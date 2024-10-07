import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from colorama import Fore, Style
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.oauth2 import service_account
from loguru import logger
from tqdm import tqdm

from google_api.packages.gservice import GService, ServiceKey

warnings.filterwarnings(
    "ignore",
    message="BigQuery Storage module not found, fetch data with the REST endpoint instead.",
    category=UserWarning,
    module="google.cloud.bigquery.table",
)

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
        self.project = project or os.getenv("BIGQUERY_DEFAULT_PROJECT")
        if not self.project:
            raise Exception("Project not specified nor found in environment variables")
        if hasattr(self, "initialized"):
            logger.info(
                f"{self.__class__.__name__} already initialized for {self.project}"
            )
            return
        self.initialized = True

        logger.info(f"Initializing {self.__class__.__name__} for {self.project}")

        if service_key_path:
            pass
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

    def beep(self, frequency: int, duration: int):
        if not self.completion_alert:
            return
        if hasattr(self, "_beep"):
            pass
        elif platform.system() == "Windows":
            import winsound

            self._beep = winsound.Beep

        elif platform.system() == "Darwin":
            self._beep = lambda *args, **kwargs: os.system('say "beep"')
        else:
            self._beep = lambda *args, **kwargs: None

        return self._beep(frequency, duration)

    def q(
        self,
        sql: str,
        project: Optional[str] = None,
        row_limit: Optional[int] = 500,
        completion_alert: bool = False,
        is_checking_row_cnt: bool = False,
    ) -> pd.DataFrame:
        project = project or self.project
        self.completion_alert = completion_alert

        job: bigquery.QueryJob = self.client.query(sql)

        if job.ddl_target_table:
            logger.success(
                f"{job.ddl_operation_performed} done to {job.ddl_target_table};\n"
            )
        elif job.dml_stats:
            logger.success(
                f"Deleted {job.dml_stats.deleted_row_count} row(s)\n"
                if job.dml_stats.deleted_row_count
                else f"Inserted {job.dml_stats.inserted_row_count} row(s)\n"
                if job.dml_stats.inserted_row_count
                else f"Updated {job.dml_stats.updated_row_count} row(s)\n"
                if job.dml_stats.updated_row_count
                else "No rows affected"
            )
        job_result: Iterator = job.result()
        if is_checking_row_cnt:
            pass
        # get row count from job_result
        elif output_row_cnt := job_result.total_rows:
            logger.success(f"Query returned {output_row_cnt:,.0f} rows")
        else:
            logger.success("Query completed without returning any rows")

        if job.destination and not is_checking_row_cnt:
            # get temp table name
            temp_table = f"{job.destination.dataset_id}.{job.destination.table_id}"
            logger.success(f"Query result stored in {temp_table}")
            output_row_cnt = self.q(
                f"SELECT count(*) FROM {temp_table}", is_checking_row_cnt=True
            ).iloc[0, 0]

        # get row count from temp table
        if not is_checking_row_cnt:
            if row_limit and output_row_cnt > row_limit:
                return pd.DataFrame(
                    [
                        {
                            "row_limit_exceeded": f"{output_row_cnt=:,.0f} exceeds {row_limit=:,.0f}"
                        }
                    ]
                )

        self.beep(440, 300)
        return job_result.to_dataframe()

    def upload_df_to_bq(
        self,
        df,
        table_id,
        dataset_id,
        replace=False,
        specify_dtypes=False,
        batched=False,
    ) -> None:
        def upload(df_slice):
            self.bq_service.upload_df_to_bq(
                df_slice,
                table_id=table_id,
                dataset_id=dataset_id,
                replace=replace,
                schema=schema,
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
                upload(df[start:end])
        else:
            schema = self.get_schema(df, specify_dtypes)
            upload(df)

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
                    print(f"{i+1}. {dtype}")
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

    def see_query_example(
        self,
        table_name: str,
        dataset_name: str,
        keywords: Optional[str] = None,
        row_cnt: int = 5,
        min_run: int = 5,
        dbda_only: bool = True,
        is_nested: bool = False,
    ) -> list[str]:
        keywords_filter = (
            f'and regexp_contains(query,r"(?i){keywords}")' if keywords else ""
        )
        dbda_filter = (
            (
                'AND ( regexp_contains(user,r"(?i)db-airflow") or regexp_contains(user_grp,r"(?i)dbda") )'
            )
            if dbda_only
            else ""
        )
        df = self.q(
            rf"""
            SELECT DISTINCT 
                query,
                COUNT(*) as cnt
            FROM dev_dbda.bq_query_history_analysis
            WHERE 1=1 
            AND (
                regexp_contains(referenced_tables,r"(?i){dataset_name}.{table_name}")
                or regexp_contains(destination_table,r"(?i){dataset_name}.{table_name}")
            )
            AND regexp_contains(job_type_level2,r"(?i)scheduled_query|script_job")
            {dbda_filter}
            {keywords_filter}
            GROUP BY All having cnt >= {min_run} 
            ORDER BY cnt DESC limit {row_cnt}
        """
        )
        if df.empty and not is_nested:
            return self.see_query_example(
                table_name=table_name,
                dataset_name=dataset_name,
                keywords=keywords,
                row_cnt=row_cnt,
                min_run=1,
                dbda_only=False,
                is_nested=True,
            )

        query_list = df["query"].tolist()

        return query_list

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

    def q_database_index(
        self,
        column_keyword: Optional[str] = None,
        table_keyword: Optional[str] = None,
        dataset_keyword: Optional[str] = None,
    ) -> pd.DataFrame:
        assert (
            self.project == "fairprice-bigquery"
        ), "Not implemented for projects other than fairprice-bigquery."
        assert (
            column_keyword or dataset_keyword or table_keyword
        ), "At least one keyword must be provided."
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
        results = self.q(
            f"""
            SELECT DISTINCT  
                table_id, 
                column_name,
                is_partitioning_column,
                data_type,
                # clustering_ordinal_position,
            FROM ads_dbda.db_database_index
            WHERE 1=1 
            {col_filter}
            {dataset_filter}
            {table_filter}
            ORDER BY 1,2,3
        """
        )
        return results

    def get_table_sample_rows(
        self,
        table_keyword: str,
        dataset_keyword: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        assert (
            self.project == "fairprice-bigquery"
        ), "Not implemented for projects other than fairprice-bigquery."
        metadata = self.q_database_index(
            table_keyword=table_keyword,
            dataset_keyword=dataset_keyword,
        )
        unique_table_ids = set(metadata["table_id"])
        assert len(unique_table_ids) <= 3, f"Too many tables found: {unique_table_ids}"
        table_partition_col_mapping: dict[str, list[str]] = {}
        for table_id in unique_table_ids:
            partitions = metadata[
                (metadata["table_id"] == table_id)
                & (metadata["is_partitioning_column"])
            ]
            table_partition_col_mapping[table_id] = partitions["column_name"].tolist()

        results = {}
        for table_id in unique_table_ids:
            first_partition_col = table_partition_col_mapping[table_id][0]
            max_partition = self.q(
                rf"""
                SELECT MAX({first_partition_col}) AS max_partition
                FROM `{table_id}`
                """
            ).iloc[0]["max_partition"]
            df = self.q(
                f"""
            SELECT * FROM `{table_id}`
            where concat({first_partition_col}) = '{max_partition}'
            limit 5
            """
            )
            results[table_id] = df.to_markdown()
        return results

    def q_last_modified(
        self,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
    ) -> pd.Timestamp:
        project = project or self.project
        results = self.q(f"""
        SELECT TIMESTAMP_MILLIS(last_modified_time) AS last_modified
        FROM `{project}.{dataset_id}.__TABLES__`,
        UNNEST([table_id = "{table_id}"]) AS is_exact_match,
        UNNEST([table_id LIKE "{table_id}%"]) AS is_like_match
        WHERE 1=1 
            AND is_like_match
        ORDER BY is_exact_match DESC, last_modified DESC
        LIMIT 1
        """)
        if results.empty:
            logger.warning(f"No results found for table {table_id}")
        return results["last_modified"][0]

    def list_table_by_keyword(
        self,
        keyword: Optional[str] = None,
        dataset_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list:
        project = project or self.project
        if dataset_id:
            dataset_ref = self.bq_service.ensure_dataset(dataset_id, project)
            tables = list(self.client.list_tables(dataset_ref))
        else:
            datasets = self.client.list_datasets()
            tables = []
            for dataset in datasets:
                dataset_ref = self.client.dataset(dataset.dataset_id, project)
                tables += list(self.client.list_tables(dataset_ref))
        table_names = [
            table.full_table_id
            for table in tables
            if not keyword or keyword in table.table_id
        ]
        logger.success(f"{len(table_names)} tables found.")
        return table_names

    def list_dataset_by_keyword(self, keyword, project=None):
        project = project or self.project
        datasets = self.client.list_datasets()
        datasets = [
            dataset.full_dataset_id
            for dataset in datasets
            if keyword in dataset.dataset_id
        ]
        logger.success(f"{len(datasets)} datasets found.")
        return datasets


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
        logger.success(
            f"initializing {self.__class__.__name__} with project {self.project}"
        )
        start_ts = time.time()
        self.credentials = service_account.Credentials.from_service_account_info(
            info=self.gservice.sa_info, scopes=self.SCOPES
        )
        self.client = bigquery.Client(project=project, credentials=self.credentials)
        duration = time.time() - start_ts
        logger.success(
            f"BigQuery client connected successfully in {duration:,.2f} seconds"
        )

    def upload_df_to_bq(
        self,
        df: pd.DataFrame,
        table_id: str,
        dataset_id: str,
        project: Optional[str] = None,
        replace: bool = False,
        schema: Optional[list[bigquery.SchemaField]] = None,
    ):  # -> "_AsyncJob"
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE" if replace else "WRITE_APPEND",
        )

        project = project or self.project
        table_ref = self.get_table(table_id, dataset_id, project)
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
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


def main():
    bq = BigQuery(project="fairprice-bigquery")
    print(bq.q("SELECT current_date('+8') as date").to_markdown())


if __name__ == "__main__":
    main()
