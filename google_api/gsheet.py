import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from googleapiclient.errors import HttpError
from loguru import logger

from google_api.packages.gservice import GService, ServiceKey


class GSheetAPI(GService):
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SCOPES_SN = "sheets"
    VERSION = "v4"
    _instances: dict = {}

    def __new__(cls, *args, **kwargs):
        spreadsheet_id = kwargs.get("spreadsheet_id")
        if spreadsheet_id not in cls._instances:
            cls._instances[spreadsheet_id] = super().__new__(cls)
        return cls._instances[spreadsheet_id]

    def __init__(
        self,
        value=None,
        range: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
        service_key: Optional[ServiceKey] = None,
        service_key_path: Optional[Path | str] = None,
        service_key_env_var: Optional[str] = None,
    ) -> None:
        logger.info("Initializing GSheetAPI")
        super().__init__(
            service_key=service_key,
            service_key_path=service_key_path,
            service_key_env_var=service_key_env_var,
        )
        self.build_service(
            scopes=self.SCOPES, short_name=self.SCOPES_SN, version=self.VERSION
        )
        if spreadsheet_id is not None:
            self.range = range or "A1"
            self.spreadsheet_id = (
                spreadsheet_id or "1VgXfe0xIj2trHh0nxedjtmIKIrt_I9NpauqjuKJukkA"
            )

        if value is not None:
            assert (
                spreadsheet_id is not None
            ), "Spreadsheet ID must be specified for value to be appended"
            self.value = value
            self.append_row(self.value, self.range, self.spreadsheet_id)
        logger.success("Initialized GSheetAPI")

    def sheet_to_df(
        self, sheet_name, spreadsheet_id=None, has_header=False
    ) -> pd.DataFrame:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        df = self.read_cells(f"'{sheet_name}'!A:ZZZ", spreadsheet_id, has_header)
        logger.success(f"Converted sheet {sheet_name} to DataFrame")
        return df

    def sheet_exists(self, sheet_name: str, create_if_not_exists=False) -> bool:
        sheets = self.get_all_sheets()
        if any(
            sheet_name == sheet["properties"]["title"] for sheet in sheets["sheets"]
        ):
            logger.info(f"Sheet {sheet_name} exists")
            return True
        log = f"Sheet {sheet_name} does not exist."
        if create_if_not_exists:
            self.create_sheet(sheet_name)
            log += " Creating sheet..."
        logger.info(log)
        return False

    def read_cells(self, range, spreadsheet_id=None, has_header=False) -> pd.DataFrame:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range,
                    valueRenderOption="UNFORMATTED_VALUE",
                )
                .execute()
            )
            values = result.get("values", [])
            self.version = result.get("developerMetadata", [{}])[0].get("metadataValue")
        except HttpError as e:
            error_message = str(e)
            if "Unable to parse range" in error_message:  # Handle the specific error
                logger.error(
                    f"HttpError: Unable to parse range {range} in {spreadsheet_id}"
                )
                return pd.DataFrame()
            else:  # Log or handle the error if it is a different HttpError
                logger.error(f"An HttpError occurred: {error_message}")
                raise
        except Exception as e:
            logger.error(f"An expected error occurred: {e}")
            raise
        logger.success(f"Read {len(values)} rows from {range} in {spreadsheet_id}")
        if has_header and len(values) > 0:
            header = values[0]
            data = values[1:]
            return pd.DataFrame(data, columns=header)
        return pd.DataFrame(values)

    def get_all_sheets(self, spreadsheet_id=None) -> dict:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        sheets = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        logger.success(f"Got all sheets from {spreadsheet_id}")
        return sheets

    def loop_delete_empty_sheets(self, spreadsheet_id=None) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        sheets = self.get_all_sheets(spreadsheet_id)
        sheet_titles = [sheet["properties"]["title"] for sheet in sheets["sheets"]]
        for sheet in sheet_titles:
            df = self.read_cells(f"'{sheet}'!A:ZZZ")
            if df.empty:
                self.delete_sheet(sheet)
                logger.success(f"Sheet {sheet} deleted")

    def _get_sheet_id(self, sheet_name, spreadsheet_id=None) -> int:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        sheets = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_titles = [sheet["properties"]["title"] for sheet in sheets["sheets"]]
        sheet_ids = [sheet["properties"]["sheetId"] for sheet in sheets["sheets"]]
        sheet_id = sheet_ids[sheet_titles.index(sheet_name)]
        logger.success(f"Got sheet ID {sheet_id} for {sheet_name}")
        return sheet_id

    def delete_sheet(self, sheet_name, spreadsheet_id=None) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        sheet_id = self._get_sheet_id(sheet_name, spreadsheet_id)
        body = {"requests": [{"deleteSheet": {"sheetId": sheet_id}}]}
        self.service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body
        ).execute()
        logger.success(f"Sheet {sheet_name} deleted")

    def append_row(self, value, range, spreadsheet_id=None, with_header=False) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        body = ""
        if isinstance(value, list) and isinstance(value[0], list):
            body = {"values": value}
        elif isinstance(value, list):
            body = {"values": [value]}
        elif isinstance(value, (int, float, dict, tuple)):
            value = str(value)
            body = {"values": [[value]]}
        elif isinstance(value, str):
            body = {"values": [[value]]}
        elif isinstance(value, pd.DataFrame):
            logger.info(f"Converting df of shape {value.shape} to list")
            pd.set_option("future.no_silent_downcasting", True)
            value_wo_null = value.fillna("")
            # value_wo_null = value_wo_null.infer_objects(copy=False)
            value_wo_null = value_wo_null.astype(str)
            body = {
                "values": (
                    ([value_wo_null.columns.tolist()] if with_header else [])
                    + value_wo_null.values.tolist()
                )
            }
        assert body, f"Body is empty for {value=} because {type(value)=}"
        result = (
            self.service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        updated = result.get("updates").get(
            "updatedCells"
        )  # check how many cells were updated
        if updated == 0:
            input("GSheet: No cells were updated. Press Enter to continue...")
        logger.success(f"Appended to {range} in {spreadsheet_id}")

    def get_cell_address_by_index(self, index: tuple) -> str:
        column = chr(index[1] + 65)
        row = index[0] + 1
        logger.success(f"Got cell address {column}{row} for index {index}")
        return f"{column}{row}"  # e.g. A1

    def find_row_by_value(
        self,
        value: str,
        range: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
    ) -> Any:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        values: pd.DataFrame = self.read_cells(range, spreadsheet_id)
        for i, row in values.iterrows():
            if value in str(row):
                logger.success(f"Found row {row} with value {value} in {range}")
                return row
        logger.warning(f"Value {value} not found in {range}")
        return None

    def find_cell_address_by_value(
        self, value, range, spreadsheet_id=None
    ) -> str | None:
        tuple = self.find_cell_index_by_value(value, range, spreadsheet_id)
        if tuple is None:
            logger.warning(f"Value {value} not found in {range}")
            return None
        result = self.get_cell_address_by_index(tuple)
        logger.success(f"Found cell address {result} for value {value}")
        return result

    def find_cell_index_by_value(
        self, value, range, spreadsheet_id=None
    ) -> tuple | None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        values = self.read_cells(range, spreadsheet_id)
        assert values is not None, f"Values are None for range {range}"
        for i, row in enumerate(values):
            if row is None:
                logger.warning(f"Row {i} is None for range {range}")
                continue
            if isinstance(row, list) and value in row:
                logger.success(f"Found value {value} in row {row}")
                return i, row.index(value)
        logger.warning(f"Value {value} not found in {range}")
        return None

    def get_cell_tuple_by_address(self, address: str) -> tuple | None:
        match = re.match(r"([a-zA-Z]+)([0-9]+)", address)
        if match is None:
            logger.error(f"Invalid address {address}")
            return None
        column = match.group(1).upper()
        row = int(match.group(2)) - 1
        result = (row, ord(column) - 65)
        logger.success(f"Got cell tuple {result} for address {address}")
        return result

    def update_cell(self, range, value, spreadsheet_id=None) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        body = {"values": [[value]]}
        result = (
            self.service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        updated = result.get("updatedCells")
        if updated == 0:
            logger.warning(f"No cells were updated in using {range} with {value}")
            return
        logger.success(f"Updated {range} with {value=} in {spreadsheet_id}")

    def create_sheet(self, title, spreadsheet_id=None) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id

        def _check_sheet(self, title):
            sheets = (
                self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            )
            sheet_titles = [sheet["properties"]["title"] for sheet in sheets["sheets"]]
            if title in sheet_titles:
                return True
            return False

        if _check_sheet(self, title):
            logger.info(f"Sheet {title} already exists")
            return
        body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
        result = (
            self.service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )
        logger.success(f"Created sheet {title} in {spreadsheet_id}")
        return result

    def clear_sheet(
        self, sheet_name=None, spreadsheet_id=None, leave_header=False
    ) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        sheet_name = sheet_name or ""
        cells = "A2:ZZZ" if leave_header else "A1:ZZZ"
        self.clear_range(f"{sheet_name}!{cells}", spreadsheet_id=spreadsheet_id)

    def clear_range(self, range, spreadsheet_id=None) -> None:
        spreadsheet_id = spreadsheet_id or self.spreadsheet_id
        body = {}
        result = (
            self.service.spreadsheets()
            .values()
            .clear(spreadsheetId=spreadsheet_id, range=range, body=body)
            .execute()
        )
        logger.success(f"Cleared range {range} in {spreadsheet_id}")
        return result


def main():
    spreadsheet_id = "1ajUslJg7o9-ENYEK03m5JoLACBAoBQaiJbjk1NllqDc"
    gsheet = GSheetAPI(spreadsheet_id=spreadsheet_id)
    print(gsheet.read_cells("'Coins'!A:ZZZ", has_header=True))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
