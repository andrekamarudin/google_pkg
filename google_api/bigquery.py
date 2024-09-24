import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from colorama import Fore, Style
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.oauth2 import service_account
from loguru import logger
from tqdm import tqdm

from google_api.packages.gservice import GService, ServiceKey


class BigQuery:
    _instances = {}

    def __new__(cls, project: Optional[str] = None, **kwargs):
        """
        Ensure singleton instance of BigQuery class for a given project.
        """
        project = project or "andrekamarudin"
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
        if hasattr(self, "initialized"):
            logger.info(
                f"{self.__class__.__name__} already initialized for {self.project}"
            )
            return
        self.initialized = True
        self.project = project or "andrekamarudin"
        logger.info(f"Initializing {self.__class__.__name__} for {self.project}")
        self.bq_service = BigQueryService(
            self,
            project=project,
            location=location,
            service_key_path=service_key_path,
            service_key=service_key,
        )
        self.client = self.bq_service.client

    def q(self, sql: str, project: Optional[str] = None) -> pd.DataFrame:
        project = project or self.project
        df = self.client.query(sql).to_dataframe()
        logger.info(
            f"Query executed: {df.shape[0]} rows and {df.shape[1]} columns; {df.memory_usage(deep=True).sum() / 1e6} MB"
        )
        return df

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
        logger.warning("Attempting to infer schema from DataFrame")
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
            logger.warning("No schema inferred")
            return schema

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

    def list_table_by_keyword(
        self,
        keyword: str,
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
            table.full_table_id for table in tables if keyword in table.table_id
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
        if hasattr(self, "initialized"):
            logger.info(
                f"{self.__class__.__name__} already initialized for {self.project}"
            )
            return
        self.initialized = True
        self.project = project or "andrekamarudin"
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
    sa_path = Path(
        r"C:\Users\andre\AppData\Roaming\Andre\dbda_google_service_account.json"
    )
    service_key = ServiceKey(**json.loads(sa_path.read_text()))
    bq = BigQuery(project="fairprice-bigquery", service_key=service_key)
    print(bq.q("SELECT * FROM dev_dbda.calendar_full").to_markdown())


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="SUCCESS")
    main()
