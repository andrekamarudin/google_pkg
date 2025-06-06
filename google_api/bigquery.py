# %%
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
from typing import Any, Optional, Set

import numpy as np
import pandas as pd
from colorama import Fore, Style
from dateutil.relativedelta import relativedelta
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator
from google.oauth2 import service_account
from loguru import logger
from tqdm import tqdm

sys.path.extend([str(x) for x in Path(__file__).parents])
from packages.gservice import GService, ServiceKey


class GoogleAPIError(Exception):
    pass


# Define warnings to filter
WARNINGS_TO_FILTER = [
    {"message": "To exit: use 'exit', 'quit', or Ctrl-D.", "category": None},
    {
        "message": "BigQuery Storage module not found, fetch data with the REST endpoint instead.",
        "category": UserWarning,
        "module": "google.cloud.bigquery.table",
    },
]
for warning in WARNINGS_TO_FILTER:
    kwargs = {k: v for k, v in warning.items() if v is not None}
    # Use regex pattern matching for messages
    if "message" in kwargs:
        kwargs["message"] = f"^{kwargs['message']}$"
    warnings.filterwarnings("ignore", **kwargs)

logger.remove()
LOG_FMT = "<level>{level}: {message}</level> <black>({file} / {module} / {function} / {line})</black>"
logger.add(sys.stdout, level="SUCCESS", format=LOG_FMT)


class BigQuery:
    _instances = {}

    def __new__(cls, project: Optional[str] = None, **kwargs):
        """
        Ensure singleton instance of BigQuery class for a given project.
        """
        project = project or os.getenv("BIGQUERY_DEFAULT_PROJECT")
        if project in cls._instances:
            return cls._instances[project]
        else:
            instance = super().__new__(cls)
            cls._instances[project] = instance
            return instance

    def __init__(
        self,
        project=None,
        location=None,
        service_key_path: Optional[Path | str] = None,
        service_key: Optional[ServiceKey] = None,
    ):
        try:
            self.project = project or os.getenv("BIGQUERY_DEFAULT_PROJECT")
            if not self.project:
                raise GoogleAPIError(
                    "Project not specified nor found in environment variables"
                )
            if hasattr(self, "initialized"):
                logger.info(
                    f"{self.__class__.__name__} already initialized for {self.project}"
                )
                return
            self.initialized = True

            logger.info(f"Initializing {self.__class__.__name__} for {self.project}")

            if service_key_path:
                pass
            elif self.project == "ne-fprt-data-cloud-production":
                service_key_path = os.getenv("CIA_GOOGLE_APPLICATION_CREDENTIALS")
            elif self.project == "fairprice-bigquery":
                service_key_path = os.getenv("DBDA_GOOGLE_APPLICATION_CREDENTIALS")
            else:
                service_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            self.bq_service = BigQueryService(
                self,
                project=project,
                location=location,
                service_key_path=service_key_path,
                service_key=service_key,
            )
            self.client = self.bq_service.client
        except Exception as e:
            self.__class__._instances.pop(self.project, None)
            raise GoogleAPIError(e)

    @property
    def dry_run_only(self) -> bool:
        """
        Check if the BigQuery instance is in dry run mode.
        """
        if not hasattr(self, "_dry_run_only"):
            self._dry_run_only = False
        return self._dry_run_only

    @dry_run_only.setter
    def dry_run_only(self, value: bool):
        """
        Set the dry run mode for the BigQuery instance.
        """
        if not isinstance(value, bool):
            raise ValueError("dry_run_only must be a boolean value.")
        self._dry_run_only = value
        if value:
            logger.warning(
                "BigQuery is set to dry run mode. No actual queries will be executed."
            )

    # async def _process_page(self, page, qbar, results):
    def _process_page(self, page, qbar, results):
        for row in page:
            results.append(dict(row))
            qbar.update(1)

    # async def _load_to_dataframe(self, job_result, qbar):
    def _load_to_dataframe(self, job_result: RowIterator):
        if job_result.total_rows and job_result.total_rows <= 100000:
            return job_result.to_dataframe()
        qbar = tqdm(
            total=job_result.total_rows, unit="rows", desc="Loading to dataframe"
        )
        results = []
        for page in job_result.pages:
            self._process_page(page, qbar, results)
        # tasks = [self._process_page(page, qbar, results) for page in job_result.pages]
        # await asyncio.gather(*tasks)
        qbar.close()
        return pd.DataFrame(results)

    def wait_for_job(
        self,
        job: bigquery.QueryJob,
        *args,
        **kwargs,
    ) -> RowIterator:
        result = job.result()
        return result

    def _query(
        self,
        sql: str,
        project: Optional[str] = None,
        page_size: int = 50000,
        sample_row_cnt: Optional[int] = 100000,
        wait_for_results: bool = True,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        project = project or self.project

        dry_run_result = self._dry_run(sql, project)
        if not dry_run_result["success"]:
            logger.warning(pformat(dry_run_result))
            return {
                "success": False,
                "error": dry_run_result["error"],
                "error_obj": dry_run_result["error_obj"],
            }
            raise GoogleAPIError(dry_run_result["error_obj"])
        sql_extract = re.sub(r"[\s\n]+", " ", sql)[:150]

        if (query_size := dry_run_result["bytes_processed"] / 1e9) > 5:
            logger.warning(f"Query bytes: {query_size:,.2f} GB")

        logger.success(sql_extract)
        start_time: float = time.time()
        job: bigquery.QueryJob = self.client.query(sql)
        if not wait_for_results:
            return {"success": True, "job": job}

        try:
            job_result: RowIterator = job.result(page_size=page_size)
            self._last_job: bigquery.QueryJob = job  # Store the job instead
            self._last_job_result: RowIterator = job_result
        except Exception as e:
            self._highlight_sql_error(e, sql)
            raise GoogleAPIError(e)
            return {"success": False, "error": str(e)}
        duration = timedelta(seconds=round(time.time() - start_time))
        message = f"'{job.statement_type}' done in {duration}. "

        # if job_result.total_rows == 0:
        #     # no rows returned
        #     return {"success": True, "message": f"{message}\nNo rows returned."}
        if job.statement_type and job.statement_type not in ["SELECT"]:
            # DDL or DML
            if row_cnt := job.num_dml_affected_rows:
                message += f"\n{row_cnt:,.0f} rows affected."
                message += f"Result stored in {job.destination}"
            logger.success(message)
            return {
                "success": True,
                "ddl_type": job.statement_type,
                "rows_affected": job.num_dml_affected_rows,
                "target_table": job.destination,
                "duration": duration,
            }
        elif job_result.total_rows is not None:
            # SELECT query
            message += f"\n{job_result.total_rows:,.0f} rows returned"
            message += (
                f" and stored in {job.destination.dataset_id}.{job.destination.table_id}\n"
                if job.destination
                else ""
            )
            logger.success(message)

        job_sample: RowIterator = job.result(max_results=sample_row_cnt)
        sample_result: list[dict] = [dict(row) for row in job_sample]

        return {
            "success": True,
            "message": message,
            "job_result": job_result,
            "total_rows": job_result.total_rows,
            "sample_result": sample_result,
            "duration": duration,
        }

    def q(
        self,
        sql: str,
        project: Optional[str] = None,
        sample_row_cnt: int = 100000,
        wait_for_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run a query and return the results as a DataFrame.
        :param sql: SQL query to run
        :param project: Project to run the query in
        :param sample_row_cnt: Number of rows to sample from the query result. Default 100k rows. If 0, return all rows.
        :param wait_for_results: Whether to wait for the query to complete
        :return: DataFrame containing the query results
        """
        if self.dry_run_only:
            dry_run_result: dict = self._dry_run(sql, project)
            return pd.DataFrame([dry_run_result])

        result: dict[str, Any] = self._query(
            sql=sql,
            project=project,
            sample_row_cnt=sample_row_cnt,
            wait_for_results=wait_for_results,
        )

        if not wait_for_results:
            return pd.DataFrame()
        elif "job_result" not in result:
            return pd.DataFrame(result, index=[0])

        total_rows = result.get("total_rows", 0)
        if sample_row_cnt == 0 or (total_rows <= sample_row_cnt):
            # unlimited rows or within limit
            final_result = self._load_to_dataframe(result["job_result"])
        else:
            #  exceeded the limit
            logger.error(
                f"Query returned {total_rows} rows, which is over the limit set. Retrieving only the first {sample_row_cnt} rows. Set sample_row_cnt=0 to retrieve all rows."
            )
            final_result = pd.DataFrame(result.get("sample_result"))

        if final_result is None:
            return pd.DataFrame(result, index=[0])
        elif final_result.empty:
            return pd.DataFrame()
        return final_result

    def full_describe(self, df: pd.DataFrame) -> dict[str, dict]:
        numeric_columns = df.select_dtypes(include=np.number).columns.to_list() or []
        non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
        return {
            "numeric_columns": (
                (df[numeric_columns].describe().to_dict()) if numeric_columns else {}
            ),
            "non_numeric_columns": (
                (df[non_numeric_columns].describe().to_dict())
                if non_numeric_columns
                else {}
            ),
        }

    def upload_df_to_bq(
        self,
        df,
        table_id,
        dataset_id,
        replace=False,
        job_config=None,
        specify_dtypes=False,
        batched=False,
        schema=None,
        silent: bool = False,
    ) -> None:
        def upload(df_slice, silent):
            self.bq_service.upload_df_to_bq(
                df_slice,
                table_id=table_id,
                dataset_id=dataset_id,
                replace=replace,
                schema=schema,
                silent=silent,
                job_config=job_config,
            )

        if batched:
            rows = len(df)
            batches = 20 if rows >= 20000 else 2
            batch_size = rows // batches
            for i in tqdm(range(batches)):
                schema = self.get_schema(df, specify_dtypes) if i == 0 else None
                start = i * batch_size
                end = start + batch_size
                replace = replace if i == 0 else False
                upload(df[start:end], silent)
        else:
            schema = schema or self.get_schema(df, specify_dtypes)
            upload(df, silent)

    def get_schema(
        self, df: pd.DataFrame, specify_dtypes: bool
    ) -> list[bigquery.SchemaField] | None:
        logger.info("Attempting to infer schema from DataFrame")
        if specify_dtypes:
            column_names = df.columns.tolist()
            common_dtypes = [
                "STRING",
                "INTEGER",
                "FLOAT",
                "DATETIME",
                "TIMESTAMP",
                "DATE",
                "TIME",
                "BOOLEAN",
            ]
            column_dtypes = {}
            for column in column_names:
                os.system("cls")
                example_values = df[column].dropna().unique()[:5].tolist()
                print(f"{Fore.YELLOW}Column: {column}")
                print("Example values: ", example_values)
                print("Please select the data type from the following options:")
                print(Fore.LIGHTBLACK_EX, end="")
                for i, dtype in enumerate(common_dtypes):
                    print(f"{i + 1}. {dtype}")
                user_input = input(
                    f"Enter the number corresponding to the data type: {Fore.GREEN}"
                )
                dtype_index = int(user_input) - 1
                column_dtypes[column] = common_dtypes[dtype_index]
                print(Style.RESET_ALL)
            schema = [
                bigquery.SchemaField(name, dtype)
                for name, dtype in column_dtypes.items()
            ]
            logger.success(f"Schema inferred successfully: {schema}")
            return schema
        else:
            schema = None
            logger.info("No schema inferred")
            return schema

    def _dry_run(self, sql: str, project: Optional[str] = None) -> dict:
        project = project or self.project
        job_config = bigquery.QueryJobConfig(dry_run=True)
        try:
            job: bigquery.QueryJob = self.client.query(sql, job_config=job_config)
            return {
                "success": True,
                "bytes_processed": job.total_bytes_processed,
                "statement_type": job.statement_type,
            }
        except Exception as e:
            self._highlight_sql_error(e, sql)
            return {
                "success": False,
                "error": str(e),
                "error_obj": e,
            }

    def _highlight_sql_error(self, error, sql: str) -> str:
        match = re.search(r"at \[(\d+):(\d+)\]", str(error))
        error_to_print = pformat(str(error))
        if match and "failed to parse view" not in error_to_print:
            line_no, char_no = match.groups()
            line_no = int(line_no) - 1
            char_no = int(char_no) - 1
            sql_lines = sql.splitlines()
            lines_to_show = 5
            lines_bef = sql_lines[max(0, line_no - lines_to_show) : line_no]
            lines_aft = sql_lines[line_no + 1 : line_no + lines_to_show]
            line = sql_lines[line_no]
            lb = "\n"
            lines_colored = (
                rf"{Fore.LIGHTYELLOW_EX}"
                rf"{lb.join(lines_bef)}{lb}"
                rf"{line[:char_no]}"
                rf"{Fore.LIGHTRED_EX}{line[char_no:]}"
                rf"{Fore.LIGHTYELLOW_EX}{lb}"
                rf"{lb.join(lines_aft)}"
                rf"{Fore.RESET}"
            )
            error_to_print += f"\n{lines_colored}\n"
        logger.error(error_to_print)
        match = re.search(r"at \[(\d+):(\d+)\]", str(error))
        return error_to_print

    def drop_dataset(self, dataset_id: str, project: Optional[str] = None) -> None:
        project = project or self.project
        self.client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
        logger.success(f"Dataset `{project}:{dataset_id}` dropped successfully")
        return

    def drop_table(
        self, table_id: str, dataset_id: str, project: Optional[str] = None
    ) -> None:
        project = project or self.project
        table_ref = self.bq_service.get_table(table_id, dataset_id, project)
        self.client.delete_table(table_ref, not_found_ok=True)
        logger.success(
            f"Table `{project}:{dataset_id}.{table_id}` dropped successfully"
        )
        return

    def get_table_schema(
        self,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
    ) -> set[str]:
        result = self._query(
            rf"""
            SELECT *
            FROM `{project}`.{dataset_id}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = '{table_id}';
            """
        )
        if not result["success"]:
            return set()
        df = self._load_to_dataframe(result["job_result"])
        tables: Set[str] = {
            f"{row['column_name']}: ({row['data_type']}){' is_partitioning_column' if row['is_partitioning_column'] == 'YES' else ''}"
            for _, row in df.iterrows()
        }
        return tables

    def search_datasets(
        self,
        dataset_keyword: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[str]:
        project = project or self.project
        datasets = self.client.list_datasets(project=project)
        return [
            dataset.dataset_id
            for dataset in datasets
            if not dataset_keyword or re.search(dataset_keyword, dataset.dataset_id)
        ]

    def search_tables(
        self,
        table_keyword: Optional[str] = None,
        dataset_keyword: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[str]:
        project = project or self.project
        dataset_id_list = self.search_datasets(dataset_keyword, project)
        return [
            f"`{project}`.{next_dataset_id}.{table.table_id}"
            for next_dataset_id in dataset_id_list
            for table in list(
                self.client.list_tables(self.client.dataset(next_dataset_id, project))
            )
            if not table_keyword or re.search(table_keyword, table.table_id)
        ]

    def delete_table_or_view(
        self,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
    ):
        project = project or self.project
        table_ref = self.client.dataset(dataset_id, project).table(table_id)
        try:
            self.client.delete_table(table_ref, not_found_ok=True)
            logger.success(f"Table or view {table_id} deleted successfully.")
        except NotFound:
            logger.warning(f"Table or view {table_id} not found.")

    def q_database_tables(
        self,
        column_keyword: Optional[str] = None,
        table_keyword: Optional[str] = None,
        dataset_keyword: Optional[str] = None,
    ) -> Set[str]:
        df = self.q_database_index(
            column_keyword=column_keyword,
            table_keyword=table_keyword,
            dataset_keyword=dataset_keyword,
        )
        return set(df["table_id"])

    # fairprice-bigquery specific
    def q_database_index(
        self,
        column_keyword: Optional[str] = None,
        table_keyword: Optional[str] = None,
        dataset_keyword: Optional[str] = None,
    ) -> pd.DataFrame:
        assert self.project == "fairprice-bigquery", (
            "Not implemented for projects other than fairprice-bigquery."
        )
        assert column_keyword or dataset_keyword or table_keyword, (
            "At least one keyword must be provided."
        )
        col_filter = (
            f"AND REGEXP_CONTAINS(column_name,r'(?i){column_keyword}')"
            if column_keyword
            else ""
        )
        dataset_filter = (
            f"AND REGEXP_CONTAINS(table_schema,r'(?i){dataset_keyword}')"
            if dataset_keyword
            else ""
        )
        table_filter = (
            f"AND REGEXP_CONTAINS(table_name,r'(?i){table_keyword}')"
            if table_keyword
            else ""
        )
        sql = f"""
            SELECT DISTINCT  
                table_id, 
                column_name,
                data_type,
                STRING_AGG(if(is_partitioning_column='YES',column_name,Null),', ')
                 over(PARTITION BY table_id) as table_partitioning_columns,
            FROM ads_dbda.db_database_index
            WHERE 1=1 
            {col_filter}
            {dataset_filter}
            {table_filter}
            ORDER BY 1,2,3
        """
        return self.q(sql)

    def q_last_modified(
        self,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
    ) -> datetime:
        project = project or self.project
        results = self.q(f"""
        SELECT TIMESTAMP_MILLIS(last_modified_time) AS last_modified
        FROM `{project}.{dataset_id}.__TABLES__`,
        UNNEST([table_id = "{table_id}"]) AS is_exact_match,
        UNNEST([table_id LIKE "{table_id}%"]) AS is_like_match
        WHERE is_like_match
        ORDER BY is_exact_match DESC, last_modified DESC
        LIMIT 1
        """)
        if results.empty:
            logger.warning(f"No results found for table {table_id}")
            return datetime(1970, 1, 1)
        elif len(results) > 1:
            logger.warning(
                f"Multiple results found for table {table_id}. Returning the most recent one."
            )
        pd_time: pd.Timestamp = results["last_modified"][0]
        return pd.to_datetime(pd_time)

    def see_scheduled_query_example(
        self,
        destination_table_name: str,
        destination_dataset_name: str,
        source_table_name: str,
        source_dataset_name: str,
        keywords: Optional[str] = None,
        row_cnt: int = 5,
        min_run: int = 5,
        dbda_only: bool = True,
        is_nested: bool = False,
    ) -> list[str]:
        table_filters = []
        table_filters.append(
            f'regexp_contains(referenced_tables,r"(?i){source_dataset_name}.{source_table_name}")'
            if source_table_name
            else None
        )
        table_filters.append(
            f'regexp_contains(destination_table,r"(?i){destination_dataset_name}.{destination_table_name}")'
            if destination_table_name
            else None
        )
        # Filter out None values
        table_filters = [filter for filter in table_filters if filter is not None]
        table_filters_str = "and " + (" and ".join(table_filters) or "1=1")
        return self.see_query_example(
            keywords=keywords,
            row_cnt=row_cnt,
            min_run=min_run,
            dbda_only=dbda_only,
            is_nested=is_nested,
            additional_filters=table_filters_str,
            table_to_use="dev_dbda.bq_query_history_analysis",
        )

    def see_query_example(
        self,
        keywords: Optional[str] = None,
        row_cnt: int = 5,
        min_run: int = 5,
        dbda_only: bool = True,
        is_nested: bool = False,
        additional_filters: str = "",
        table_to_use: str = "dev_dbda.bq_query_history",
    ) -> list[str]:
        keywords_filter = (
            f'and regexp_contains(query,r"(?i){keywords}")' if keywords else ""
        )
        dbda_filter = (
            'AND ( regexp_contains(user,r"(?i)db-airflow") or regexp_contains(user_grp,r"(?i)dbda") )'
            if dbda_only
            else ""
        )
        df = self.q(
            rf"""
                SELECT DISTINCT 
                    query,
                    COUNT(*) as cnt
                FROM {table_to_use}
                WHERE 1=1 
                {dbda_filter}
                {keywords_filter}
                {additional_filters}
                AND creation_date >= date_add(current_date("+8"),interval -2 month)
                AND regexp_contains(job_type_level2,r"(?i)scheduled_query|script_job")
                GROUP BY All having cnt >= {min_run} 
                ORDER BY cnt DESC limit {row_cnt}
            """
        )
        if df.empty and not is_nested:
            return self.see_query_example(
                keywords=keywords,
                row_cnt=row_cnt,
                min_run=1,
                dbda_only=False,
                is_nested=True,
                additional_filters=additional_filters,
                table_to_use=table_to_use,
            )
        if "query" not in df.columns:
            return [df.to_markdown()]

        return df["query"].tolist()

    def indent_query(self, query: str, indent: str = "    ") -> str:
        return "".join(indent + line for line in query.splitlines(keepends=True))


class BigQueryService:
    _instances = {}

    def __new__(
        cls,
        bq_wrapper: BigQuery,
        project: Optional[str] = None,
        **kwargs,
    ):
        project = project or "andrekamarudin"
        if project in cls._instances:
            return cls._instances[project]
        else:
            instance = super().__new__(cls)
            cls._instances[project] = instance
            return instance

    def __init__(
        self,
        bq_wrapper: BigQuery,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_key_path: Optional[Path | str] = None,
        service_key: Optional[ServiceKey] = None,
    ):
        self.project = project or os.getenv("BIGQUERY_DEFAULT_PROJECT")
        if hasattr(self, "initialized"):
            logger.info(
                f"{self.__class__.__name__} already initialized for {self.project}"
            )
            return
        self.initialized = True
        self.location = location or "asia-southeast1"
        self.bq_wrapper = bq_wrapper
        self.SCOPES = ["https://www.googleapis.com/auth/bigquery"]
        self.gservice = GService(
            service_key_path=service_key_path, service_key=service_key
        )
        logger.info(
            f"initializing {self.__class__.__name__} with project {self.project}"
        )
        start_ts = time.time()
        self.credentials = service_account.Credentials.from_service_account_info(
            info=self.gservice.sa_info, scopes=self.SCOPES
        )
        self.client = bigquery.Client(project=project, credentials=self.credentials)
        duration = time.time() - start_ts
        logger.success(f"BigQuery {self.project} connected in {duration:,.2f} seconds")

    def upload_df_to_bq(
        self,
        df: pd.DataFrame,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
        replace: bool = False,
        silent: bool = False,
        schema: Optional[list[bigquery.SchemaField]] = None,
        job_config: Optional[bigquery.LoadJobConfig] = None,
    ):  # -> "_AsyncJob"
        job_config = job_config or bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE" if replace else "WRITE_APPEND",
        )

        project = project or self.project
        table_ref = self.get_table(table_id, dataset_id, project)
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        if not silent:
            logger.success(
                f"Uploaded to {project}:{dataset_id}.{table_id}: {df.shape[0]} rows and {df.shape[1]} columns; {df.memory_usage(deep=True).sum() / 1e6} MB"
            )

        return job.result()

    def ensure_dataset(
        self,
        dataset_id: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> bigquery.Dataset:
        project = project or self.project
        location = location or self.location
        dataset_ref = self.client.dataset(dataset_id, project=project)
        try:
            dataset = self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_id} exists")
        except NotFound:
            dataset_info = bigquery.Dataset(dataset_ref)
            dataset_info.location = location
            dataset = self.client.create_dataset(dataset_info)
            logger.info(f"Created dataset {dataset_id} in {location}.")
        return dataset

    def get_table(
        self,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
    ) -> bigquery.Table | bigquery.TableReference:
        project = project or self.project
        dataset = self.ensure_dataset(dataset_id, project)
        table_ref = dataset.table(table_id)
        try:
            table = self.client.get_table(table_ref)
            logger.info(f"Table {table_id} exists")
            return table
        except NotFound:
            logger.error(f"Table {table_id} does not exist. Continuing in 5 seconds...")
            return table_ref

    def get_running_jobs(self) -> list[bigquery.job.QueryJob]:
        return list(
            self.client.list_jobs(
                # state_filter="RUNNING",
                all_users=False,
                min_creation_time=datetime.now() - relativedelta(days=1),
                page_size=10,
            )
        )


def main():
    bq = BigQuery(project="fairprice-bigquery")
    print(bq.bq_service.get_running_jobs())


if __name__ == "__main__":
    logger.remove()
    LOG_FMT = "<level>{level}: {message}</level> <black>({file} / {module} / {function} / {line})</black>"
    logger.add(sys.stdout, level="SUCCESS", format=LOG_FMT)
    main()
