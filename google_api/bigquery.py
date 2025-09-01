# %%
import os
import re
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from textwrap import indent as _indent
from typing import Any, Optional, Set

import numpy as np
import pandas as pd
from colorama import Fore, Style
from dateutil.relativedelta import relativedelta
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud.bigquery.job.base import _AsyncJob
from google.cloud.bigquery.table import RowIterator
from google.oauth2 import service_account
from loguru import logger
from tqdm import tqdm

sys.path.extend([str(x) for x in Path(__file__).parents])
from packages.charting import chart
from packages.gservice import GService

indent = partial(_indent, prefix="    ")


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
        service_key: Optional[dict] = None,
        dry_run_only: bool = False,
    ):
        try:
            self.project: str = project or os.environ["BIGQUERY_DEFAULT_PROJECT"]
            if not self.project:
                logger.error("Project not specified nor found in environment variables")
                raise SystemExit()
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
            if service_key_path:
                assert Path(service_key_path).exists(), (
                    f"Service key path {service_key_path} does not exist."
                )
            else:
                assert service_key, (
                    "Either service_key_path or service_key must be provided."
                )
            self.bq_service = BigQueryService(
                self,
                project=project,
                location=location,
                service_key_path=service_key_path,
                service_key=service_key,
            )
            self.client = self.bq_service.client
            self.chart = staticmethod(chart)
            self.dry_run_only: bool = dry_run_only
            logger.success(f"BigQuery {self.project} connected")
        except Exception as e:
            self.__class__._instances.pop(self.project, None)
            logger.error(e)
            raise SystemExit()

    def _load_to_dataframe(self, job_result: RowIterator) -> pd.DataFrame:
        if job_result.total_rows and job_result.total_rows <= 100000:
            return job_result.to_dataframe()

        qbar = tqdm(
            total=job_result.total_rows,
            unit=" rows",
            desc="Load to df",
        )
        results: list[dict] = []
        for page in job_result.pages:
            results += [dict(row) for row in page]
            qbar.n = len(results)
            qbar.refresh()
        return pd.DataFrame(results)

    def wait_for_job(self, job: bigquery.QueryJob, *args, **kwargs) -> RowIterator:
        result = job.result()
        return result

    def _query(
        self,
        sql: str,
        project: Optional[str] = None,
        page_size: int = 50000,
        sample_row_cnt: Optional[int] = 100000,
        wait_for_results: bool = True,
        skip_dry_run: bool = False,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        project = project or self.project

        if not skip_dry_run:
            dry_run_result = self._dry_run(sql, project)
            if not dry_run_result.get("success", False):
                return dry_run_result
            query_size: float = dry_run_result.get("bytes_processed", 0) / 1e9
            if query_size > 5:
                logger.warning(f"Query bytes: {query_size:,.2f} GB")

        sql_extract = re.sub(r"[\s\n]+", " ", sql)[:150]
        logger.success(sql_extract)
        self._last_sql = sql  # Store the last SQL query for reference

        start_time: float = time.time()
        job: bigquery.QueryJob = self.client.query(sql)
        if not wait_for_results:
            return {"job": job, "job_id": job.job_id}

        try:
            job_result: RowIterator = job.result(page_size=page_size)
            self._last_job: bigquery.QueryJob = job  # Store the job instead
            self._last_job_result: RowIterator = job_result
        except Exception as e:
            self._show_error(e, sql)
            logger.error(e)
            raise SystemExit()
        duration = timedelta(seconds=round(time.time() - start_time))
        message = f"'{job.statement_type}' done in {duration}. "
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
                "job_id": job.job_id,
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

        return {
            "success": True,
            "message": message,
            "job_result": job_result,
            "total_rows": job_result.total_rows,
            "job_sample": job_sample,
            "duration": duration,
            "job_id": job.job_id,
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
            dry_run_result: dict[str, Any] = self._dry_run(sql, project)
            return pd.DataFrame([dry_run_result])

        result_dict: dict[str, Any] = self._query(
            sql=sql,
            project=project,
            sample_row_cnt=sample_row_cnt,
            wait_for_results=wait_for_results,
        )

        if not result_dict.get("success", False):
            raise sys.exit(str(result_dict))

        result_df = pd.DataFrame([result_dict])
        if result_dict.get("total_rows", 0) == 0:
            logger.warning("\n" + result_df.T.to_markdown(index=False))
            return pd.DataFrame()
        elif not wait_for_results or ("job_result" not in result_dict):
            # async or if the query was not a SELECT statement
            return result_df.T if len(result_dict) == 1 else result_df

        total_rows: int = result_dict["total_rows"]
        if sample_row_cnt == 0 or (total_rows <= sample_row_cnt):
            # unlimited rows or within limit
            data_df: pd.DataFrame = self._load_to_dataframe(result_dict["job_result"])
        else:
            # exceeded the limit
            logger.error(
                f"Query returned {total_rows} rows, which is over the limit set. Retrieving only the first {sample_row_cnt} rows. Set sample_row_cnt=0 to retrieve all rows."
            )
            job_sample: RowIterator = result_dict["job_sample"]
            sample_result: list[dict] = [
                dict(row)
                for row in tqdm(job_sample, desc=f"Loading {sample_row_cnt:,.0f} rows")
            ]
            data_df: pd.DataFrame = pd.DataFrame(sample_result)

        if data_df.empty:
            return pd.DataFrame()
        return data_df

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
        project: Optional[str] = None,
        replace: bool = False,
        job_config: Optional[bigquery.LoadJobConfig] = None,
        batched: bool = False,
        schema: list[bigquery.SchemaField] | None = None,
        silent: bool = False,
        request_schema: bool = False,
    ) -> None:
        if self.dry_run_only:
            logger.warning("Dry run only mode is enabled. Skipping upload to BigQuery.")
            return
        project = project or self.project
        if not schema and request_schema:
            schema = schema or self.get_schema(df=df)
        upload: partial[_AsyncJob] = partial(
            self.bq_service.upload_df_to_bq,
            table_id=table_id,
            dataset_id=dataset_id,
            project=project,
            replace=replace,
            silent=silent,
            schema=schema,
            job_config=job_config,
        )

        if batched:
            rows: int = len(df)
            batches: int = 20 if rows >= 20000 else 2
            batch_size: int = rows // batches
            for i in tqdm(range(batches)):
                start: int = i * batch_size
                end: int = start + batch_size
                replace = replace if i == 0 else False
                upload(df=df[start:end])
        else:
            upload(df=df)

    def get_schema(self, df: pd.DataFrame) -> list[bigquery.SchemaField]:
        logger.info("Attempting to infer schema from DataFrame")
        column_names: list[str] = df.columns.tolist()
        common_dtypes: dict[str, str] = {
            "STRING": "str",
            "INTEGER": "int",
            "FLOAT": "float",
            "DATETIME": "datetime",
            "TIMESTAMP": "timestamp",
            "DATE": "date",
            "TIME": "time",
            "BOOLEAN": "bool",
        }
        column_dtypes: dict[str, str] = {}
        for column in column_names:
            os.system("cls")
            example_values = df[column].dropna().unique()[:5].tolist()
            print(f"{Fore.YELLOW}Column: {column}")
            print("Example values: ", example_values)
            print("Please select the data type from the following options:")
            print(Fore.LIGHTBLACK_EX, end="")
            for i, dtype in enumerate(common_dtypes.keys()):
                print(f"{i + 1}. {dtype}")
            user_input: str = input(
                f"Enter the number corresponding to the data type: {Fore.GREEN}"
            )
            if user_input.isdigit() and (1 <= int(user_input) <= len(common_dtypes)):
                dtype_index: int = int(user_input) - 1
                matched_type = list(common_dtypes.keys())[dtype_index]
            else:
                matched_type: str = next(
                    (
                        bq_type
                        for bq_type, py_type in common_dtypes.items()
                        if example_values and py_type in str(type(example_values[0]))
                    ),
                    "STRING",
                )

            column_dtypes[column] = matched_type
            print(Style.RESET_ALL)

        schema: list[bigquery.SchemaField] = [
            bigquery.SchemaField(name=name, field_type=dtype)
            for name, dtype in column_dtypes.items()
        ]
        logger.success(f"Schema inferred successfully: {schema}")
        return schema

    def _dry_run(self, sql: str, project: Optional[str] = None) -> dict[str, Any]:
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
            self._show_error(error=e, sql=sql)
            return {"success": False, "error": str(e)}

    def _show_error(self, error, sql: str) -> str:
        match: re.Match[str] | None = re.search(r"at \[(\d+):(\d+)\]", str(error))
        error_to_print: str = re.sub(
            pattern=r".*http\S+: (.+?)",
            repl="\\1",
            string=str(error),
        )
        if match and "failed to parse view" not in error_to_print:
            line_no_str, char_no_str = match.groups()
            line_no: int = int(line_no_str) - 1
            char_no: int = int(char_no_str) - 1
            sql_lines: list[str] = sql.splitlines()
            lines_to_show = 5
            lines_bef: list[str] = sql_lines[max(0, line_no - lines_to_show) : line_no]
            lines_aft: list[str] = sql_lines[line_no + 1 : line_no + lines_to_show]
            line: str = sql_lines[line_no]
            lb: str = "\n"
            lines_colored: str = (
                rf"{Fore.LIGHTYELLOW_EX}{lb}"
                r"```"
                rf"{lb.join(lines_bef)}{lb}"
                rf"{line[:char_no]}{Fore.LIGHTRED_EX}{line[char_no:]}{Fore.LIGHTYELLOW_EX}{lb}"
                rf"{lb.join(lines_aft)}"
                r"```"
                rf"{Fore.RESET}"
            )
            logger.error(f"\n{error_to_print}\n{lines_colored.lstrip(lb)}")
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

    def search_columns(
        self,
        column_keyword: str = ".",
        table_keyword: str = ".",
        dataset_keyword: str = ".",
        project: Optional[str] = None,
    ) -> pd.DataFrame:
        project = project or self.project
        sql = """
            SELECT 
                table_schema,
                table_name, 
                ordinal_position,
                column_name, 
                data_type, 
                is_partitioning_column, 
                TIMESTAMP_MILLIS(last_modified_time) AS table_last_modified,
            FROM `{this_project}.{this_dataset}.INFORMATION_SCHEMA.COLUMNS`
            LEFT JOIN `{this_project}.{this_dataset}.__TABLES__` on table_name = table_id
            where 1=1
            and regexp_contains(table_name,r"(?i){this_table_keyword}")
            and regexp_contains(column_name,r"(?i){this_column_keyword}")
        """
        sqls: str = " union all ".join(
            sql.format(
                this_project=project,
                this_dataset=dataset,
                this_table_keyword=table_keyword,
                this_column_keyword=column_keyword,
            )
            for dataset in self.search_datasets(
                dataset_keyword=dataset_keyword, project=project
            )
        )
        df = self.q(
            sqls + " ORDER BY table_last_modified desc, 1,2,3",
            project=project,
            sample_row_cnt=0,
        )
        if df.empty:
            logger.warning(
                f"No columns found for keyword '{column_keyword}' in tables matching '{table_keyword}' in datasets matching '{dataset_keyword}'"
            )
        return df

    def search_tables(
        self,
        table_keyword: Optional[str] = None,
        dataset_keyword: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[str]:
        project = project or self.project
        dataset_id_list: list[str] = self.search_datasets(dataset_keyword, project)
        return [
            f"`{project}`.{next_dataset_id}.{table.table_id}"
            for next_dataset_id in dataset_id_list
            for table in list(
                self.client.list_tables(self.client.dataset(next_dataset_id, project))
            )
            if not table_keyword or re.search(table_keyword, table.table_id)
        ]

    def drop_table_or_view(
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
        df: pd.DataFrame = self.q(
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

    def show_schema(
        self,
        full_table_id: str = "",
        table_id: str = "",
        dataset_id: str = "",
        sql="",
        project: str = "",
    ) -> pd.DataFrame:
        project = project or self.project
        if sql:
            self._dry_run(sql)
            inner_q = f"({sql})"
            return self.q(rf" SELECT * FROM {inner_q} LIMIT 1 ").T

        if full_table_id:
            parts = full_table_id.split(".")
            if len(parts) == 3:
                project, dataset_id, table_id = parts
            elif len(parts) == 2:
                dataset_id, table_id = parts
            else:
                raise ValueError(
                    "full_table_id must be in the format 'project.dataset.table' or 'dataset.table'."
                )
        elif not (table_id and dataset_id):
            raise ValueError(
                "Either full_table_id or dataset_id.table_id, or sql must be provided."
            )

        return self.search_columns(
            column_keyword=".",
            table_keyword=f"^{table_id}$",
            dataset_keyword=f"^{dataset_id}$",
            project=project,
        )


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
        service_key: Optional[dict] = None,
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
        self.credentials = service_account.Credentials.from_service_account_info(
            info=self.gservice.service_key, scopes=self.SCOPES
        )
        self.client = bigquery.Client(project=project, credentials=self.credentials)

    def _normalize_datetimes_for_bq(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize datetime columns for BigQuery compatibility.
        Why? Pandas uses nanosecond datetime precision (datetime64[ns]). BigQuery stores microseconds. PyArrow refuses to silently downcast, so it throws: “Casting from timestamp[ns] to timestamp[us] would lose data.”
        """
        df = df.copy()
        # 1) Make tz-aware columns UTC, then drop tz info, since BigQuery TIMESTAMP is UTC based.
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

        # 2) Round nanoseconds to microseconds, then cast to microsecond precision.
        dt_cols: pd.Index[str] = df.select_dtypes(include=["datetime64[ns]"]).columns
        for col in dt_cols:
            df[col] = df[col].dt.round("us").astype("datetime64[us]")
        return df

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
    ) -> _AsyncJob:
        project = project or self.project
        df = self._normalize_datetimes_for_bq(df)
        job_config = job_config or bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE" if replace else "WRITE_APPEND",
        )

        table_ref: bigquery.Table | bigquery.TableReference = self.get_table(
            table_id=table_id, dataset_id=dataset_id, project=project
        )
        job: bigquery.LoadJob = self.client.load_table_from_dataframe(
            dataframe=df, destination=table_ref, job_config=job_config
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
        dataset: bigquery.Dataset = self.ensure_dataset(dataset_id, project)
        table_ref: bigquery.TableReference = dataset.table(table_id)
        try:
            table: bigquery.Table = self.client.get_table(table_ref)
            logger.info(f"Table {table_id} exists")
            return table
        except NotFound:
            logger.error(f"Table {table_id} does not exist. Continuing in 5 seconds...")
            return table_ref

    def get_running_jobs(self) -> list[bigquery.QueryJob]:
        return list(
            self.client.list_jobs(
                # state_filter="RUNNING",
                all_users=False,
                min_creation_time=datetime.now() - relativedelta(days=1),
                page_size=10,
            )
        )

    def cancel_job(self, job_id: str, project: Optional[str] = None) -> _AsyncJob:
        project = project or self.project
        return self.client.cancel_job(job_id, project=project)

    def chart(
        self,
        df: pd.DataFrame,
        x_col: str = "",
        y_col: str = "",
        color_col: str = "",
        agg: str = "",
        chart_type: str = "",
    ):
        """
        Start a Gradio app to visualize a DataFrame with Plotly charts.
        """
        return chart(
            df=df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            agg=agg,
            chart_type=chart_type,
        )


def main():
    bq = BigQuery(project="andrekamarudin")

    return bq.q(
        "SELECT * FROM `andrekamarudin.ura_ods.txn_clean` order by price desc, contract_date desc, floor",
        sample_row_cnt=0,
    )


if __name__ == "__main__":
    main()
