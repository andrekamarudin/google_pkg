import os
from pathlib import Path
from pprint import pprint
from typing import Optional

import pandas as pd

from google_api.packages.gservice import GService


class GCal(GService):
    """
    A class representing Google Calendar operations.

    Attributes:
        file_name (str): The name of the Google service account JSON file.
        file_path (str): The path to the Google service account JSON file.
        sa_info_str (str): The content of the Google service account JSON file.
        sa_info (dict): The parsed JSON content of the Google service account JSON file.
        credentials (google.auth.service_account.Credentials): The credentials for accessing the Google Calendar API.
        service (googleapiclient.discovery.Resource): The Google Calendar API service.

    Methods:
        new_event: Creates a new event in the specified calendar.
        modify_event: Modifies an existing event in the specified calendar.

    """

    DEFAULT_CAL_ID = "andre.kamarudin@gmail.com"
    DEFAULT_LOCATION = "Signapore"
    DEFAULT_TIMEZONE = "Asia/Singapore"
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    SCOPES_SN = "calendar"
    VERISON = "v3"

    def __init__(
        self,
        service_key_path: Optional[Path] = None,
        service_key: Optional[dict] = None,
    ):
        """
        Initializes the GCal object.

        Args:
            service_account_json (str, optional): The name of the Google service account JSON file. Defaults to None.

        """
        super().__init__(
            service_key_path=service_key_path,
            service_key=service_key,
        )
        self.build_service(
            scopes=self.SCOPES, short_name=self.SCOPES_SN, version=self.VERISON
        )

    def new_event(
        self,
        summary,
        start_dt=None,
        end_dt=None,
        cal_id=None,
        location=None,
        description=None,
        timezone=None,
    ):
        """
        Creates a new event in the specified calendar.

        Args:
            cal_id (str): The ID of the calendar where the event will be created.
            summary (str): The summary or title of the event.
            start_dt (str, optional): The start date and time of the event in ISO 8601 format. Defaults to current time.
            end_dt (str, optional): The end date and time of the event in ISO 8601 format. Defaults to 1 hour after start time.
            location (str, optional): The location of the event. Defaults to an empty string.
            description (str, optional): The description of the event. Defaults to an empty string.
            timezone (str, optional): The timezone of the event. Defaults to "Asia/Singapore".

        Returns:
            str: The URL of the created event.

        """
        start_dt = start_dt or pd.to_datetime("now").isoformat()
        end_dt = end_dt or (pd.to_datetime("now") + pd.Timedelta("1h")).isoformat()
        cal_id = cal_id or self.DEFAULT_CAL_ID
        location = location or self.DEFAULT_LOCATION
        timezone = timezone or self.DEFAULT_TIMEZONE

        event = {
            "summary": summary,
            "location": location,
            "description": description,
            "start": {"dateTime": start_dt, "timeZone": timezone},
            "end": {"dateTime": end_dt, "timeZone": timezone},
            "recurrence": [
                # "RRULE:FREQ=DAILY;COUNT=2"
            ],
            "attendees": [
                [
                    {"email": "andre.kamarudin@gmail.com"},
                ]
            ],
            "reminders": {
                "useDefault": True,
                # "overrides": [ {"method": "email", "minutes": 24 * 60}, {"method": "popup", "minutes": 10}, ],
            },
        }
        event = self.service.events().insert(calendarId=cal_id, body=event).execute()
        return event.get("htmlLink")

    def modify_event(
        self,
        event_id,
        summary=None,
        start_dt=None,
        end_dt=None,
        location=None,
        description=None,
        timezone=None,
    ):
        """
        Modifies an existing event in the specified calendar.

        Args:
            event_id (str): The ID of the event to be modified.
            summary (str, optional): The new summary or title of the event. Defaults to None.
            start_dt (str, optional): The new start date and time of the event in ISO 8601 format. Defaults to None.
            end_dt (str, optional): The new end date and time of the event in ISO 8601 format. Defaults to None.
            location (str, optional): The new location of the event. Defaults to None.
            description (str, optional): The new description of the event. Defaults to None.
            timezone (str, optional): The new timezone of the event. Defaults to None.

        Returns:
            str: The URL of the modified event.

        """
        event = (
            self.service.events()
            .get(calendarId=self.DEFAULT_CAL_ID, eventId=event_id)
            .execute()
        )

        if summary:
            event["summary"] = summary
        if start_dt:
            event["start"]["dateTime"] = start_dt
        if end_dt:
            event["end"]["dateTime"] = end_dt
        if location:
            event["location"] = location
        if description:
            event["description"] = description
        if timezone:
            event["start"]["timeZone"] = timezone
            event["end"]["timeZone"] = timezone

        updated_event = (
            self.service.events()
            .update(calendarId=self.DEFAULT_CAL_ID, eventId=event_id, body=event)
            .execute()
        )
        return updated_event.get("htmlLink")

    def search_event(self, keyword, calendar_id=None):
        assert isinstance(keyword, str), "Keyword must be an non-empty string."
        calendar_id = calendar_id or self.DEFAULT_CAL_ID
        events = self.service.events().list(calendarId=calendar_id).execute()
        print(f"Returned {len(events['items'])} events.")
        for event in events["items"]:
            if keyword.lower() in event["summary"].lower():
                return event


def test():
    gcal = GCal()
    SUMMARY = "Test Event"
    new_event_url = gcal.new_event(SUMMARY)
    os.system(f'start "" "{new_event_url}"')


if __name__ == "__main__":
    gcal = GCal()
    keyword = "Lunch"
    event = gcal.search_event(keyword)
    pprint(event)
