# Google API Utilities

This project provides convenience wrappers around several Google APIs. It includes helpers for BigQuery, Sheets, Calendar, Gmail and Google Chat. The package is geared towards scripting and small automation tasks.

## Structure

```
google_api/
    bigquery.py     # BigQuery helpers
    gcal.py         # Google Calendar client
    gchat.py        # Send Google Chat messages
    gmail.py        # Gmail utilities
    gsheet.py       # Google Sheets client
    packages/
        gservice.py # Base service authentication
        charting.py # Simple plotting with Gradio/Plotly
        helper.py   # Misc helper functions
```

`__init__.py` exposes the main classes for easy importing.

## Installation

1. Ensure Python 3.10+ is installed.
2. Install dependencies using `pip` or [Poetry](https://python-poetry.org/).

```bash
pip install -e .            # from the repository root
# or
poetry install
```

Most classes require a Google service account key. Point the `DBDA_GOOGLE_APPLICATION_CREDENTIALS` environment variable to your JSON key or pass the path when creating a service object.

## Usage Example

```python
from google_api import BigQuery, GSheetAPI

# BigQuery example
bq = BigQuery(project="my-project")
result_df = bq.q("SELECT 1 AS demo")
print(result_df)

# Google Sheets example
sheet = GSheetAPI(spreadsheet_id="your-sheet-id")
print(sheet.read_cells("A1:B5"))
```

The modules provide many additional helper methods. Browse the source for more details.

