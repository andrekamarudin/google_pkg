import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd
import pytz
import requests
from loguru import logger

from google_api.packages.helper import condense_text


@dataclass
class TypedList(List):
    def __init__(self, item_type: type, *args):
        super().__init__()
        self.item_type = item_type
        for arg in args:
            self.append(arg)

    def append(self, item):
        if not item.__class__.__name__ == self.item_type.__name__:
            raise Exception(
                f"Func append: Cannot add {item.__class__.__name__!s} as a member to {self.__class__.__name__!s}"
            )
        super().append(item)

    def __add__(self, other):
        for arg in other:
            self.append(arg)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class Button(dict):
    pass


class Section(dict):
    pass


class Buttons(TypedList):
    def __init__(self, *args):
        super().__init__(Button, *args)


class Sections(TypedList):
    def __init__(self, *args):
        super().__init__(Section, *args)


class GoogleChat:
    DAG_VERSION = "20240729_1635"

    @staticmethod
    def _generate_user_tag(user_id):
        return f"<users/{user_id}>"

    @staticmethod
    def generate_hyperlink(text, url):
        return f"<{url}|{text}>"

    @staticmethod
    def text_to_section(text: str, header: str = "") -> Section:
        return Section(
            {
                "header": header or "",
                "widgets": [{"textParagraph": {"text": text}}],
            }
        )

    @staticmethod
    def df_to_sections(df: pd.DataFrame, index_col: Optional[str] = None) -> Sections:
        if df.empty:
            return Sections(GoogleChat.text_to_section("No data"))
        if index_col:
            df = df.set_index(index_col)

        sections = Sections()
        for index, row in df.iterrows():
            sections.append(
                GoogleChat.text_to_section(
                    text="\n".join(
                        f"{f'{col}: ' if col else ''}{val}"
                        for col, val in row.items()
                        if isinstance(val, (str, int, float))
                    ),
                    header=str(index),
                )
            )
        return sections

    @staticmethod
    def hyperlink_to_section(
        url: str, text: str = "URL", header: str = "", icon: str = "link"
    ) -> Section:
        buttons: Buttons = Buttons(
            GoogleChat.hyperlink_to_button(url=url, text=text, icon=icon)
        )
        section: Section = GoogleChat.buttons_to_section(buttons, header)
        return section

    @staticmethod
    def buttons_to_section(buttons: Buttons, header: Optional[str] = None) -> Section:
        if not isinstance(buttons, Buttons):
            raise Exception(f"Func buttons_to_section: Not a Buttons obj {buttons=}")
        return Section(
            {
                "header": header or "",
                "widgets": [{"buttonList": {"buttons": buttons}}],
            }
        )

    @staticmethod
    def text_to_section_fancy(
        header: str = "",
        top_label: str = "",
        text: str = "",
        bottom_label: str = "",
        icon: str = "DESCRIPTION",
        # control_type: str = None,
    ) -> Section:
        decorated_text = {
            "decoratedText": {
                "icon": {"materialIcon": {"name": icon}},
                "topLabel": top_label,
                "text": text,
                "bottomLabel": bottom_label,
                # "switchControl": { "name": "checkbox1", "selected": False, "controlType": control_type or "CHECKBOX", },
            }
        }
        return Section(
            {
                "header": header or "",
                "widgets": [decorated_text],
            }
        )

    @staticmethod
    def split_sections_to_columns(sections: Sections, header: str = "") -> Sections:
        """
        Convert a list of sections to a single section with multiple columns.
        """
        if isinstance(sections, list) and not isinstance(sections, Sections):
            sections = Sections(*sections)
        elif not isinstance(sections, Sections):
            raise Exception(
                f"Func split_sections_to_columns: Not a Sections obj {sections=}"
            )
        sections_w_col = Sections()
        for i in range(0, len(sections), 2):
            column_items = [
                {
                    "horizontalSizeStyle": "FILL_AVAILABLE_SPACE",
                    "horizontalAlignment": "START",
                    "verticalAlignment": "TOP",
                    "widgets": sections[i]["widgets"],
                },
            ]
            if i + 1 < len(sections):
                column_items.append(
                    {
                        "widgets": sections[i + 1]["widgets"],
                    }
                )

            sections_w_col.append(
                Section(
                    {
                        "header": header if i == 0 else "",
                        "widgets": [{"columns": {"columnItems": column_items}}],
                    }
                )
            )

        return sections_w_col

    @staticmethod
    def hyperlink_to_image_section(
        url: str,
        alt_text: str = "",
        header: str = "",
    ) -> Section:
        return Section(
            {
                "header": header,
                "widgets": [
                    {
                        "image": {
                            "imageUrl": url,
                            "altText": alt_text,
                        }
                    }
                ],
            }
        )

    @staticmethod
    def hyperlink_to_button(
        url: str, text: Optional[str] = None, icon: Optional[str] = None
    ) -> Button:
        return Button(
            {
                "text": text,
                "icon": {
                    "materialIcon": {"name": icon or "DESCRIPTION"},
                    "altText": text,
                },
                "onClick": {"openLink": {"url": url}},
            }
        )

    @staticmethod
    def _sections_to_payload(
        sections: Sections, title: str, card_id: str, subtitle: str
    ):
        payload = {
            "cardsV2": [
                {
                    "cardId": card_id,
                    "card": {
                        "header": {
                            "title": title,
                            "subtitle": subtitle,
                            "imageUrl": "https://cdn-icons-png.flaticon.com/512/6208/6208161.png",
                            "imageType": "CIRCLE",
                            "imageAltText": "DBDA Bot",
                        },
                        "sections": list(sections),
                    },
                }
            ]
        }
        return payload

    @staticmethod
    def exception_to_sections(exception: Exception, instance_url: str) -> Sections:
        header = f"Type of error raised: {type(exception).__name__}"
        exception_message: Union[str, Exception, pd.DataFrame, Sections, Section] = (
            exception.args[0]
            if isinstance(exception, Exception) and exception.args
            else exception
        )

        if isinstance(exception_message, pd.DataFrame):
            sections: Sections = GoogleChat.df_to_sections(
                exception_message, index_col=None
            )
        elif exception_message.__class__.__name__ == "Sections" or isinstance(
            exception_message, Sections
        ):
            sections: Sections = (
                exception_message
                if isinstance(exception_message, Sections)
                else Sections(exception_message)
                if isinstance(exception_message, Section)
                else Sections(GoogleChat.text_to_section(exception_message))
                if isinstance(exception_message, str)
                else Sections(GoogleChat.text_to_section(str(exception_message)))
            )
        elif exception_message.__class__.__name__ == "Section":
            sections = Sections(exception_message)
        elif exception_message.__class__.__name__ == "list":
            try:
                sections = Sections(*exception_message)
            except Exception as e:
                logger.warning(
                    f"Unexpected error when testing if exception is a list of Sections: {e}"
                )

        if "sections" not in locals():
            exception_extract = condense_text(str(exception_message))
            sections = Sections(
                GoogleChat.text_to_section(exception_extract, header=header)
            )

        button: Section = GoogleChat._instance_button(instance_url)
        sections.append(button)
        return sections

    @staticmethod
    def _instance_button(instnace_url: str) -> Section:
        button: Button = GoogleChat.hyperlink_to_button(
            url=instnace_url, text="Find instance", icon="search"
        )
        buttons: Buttons = Buttons(button)
        section: Section = GoogleChat.buttons_to_section(buttons)
        return section

    def send_gchat(
        self,
        message,
        webhook: str = "",
        footer: str = "",
        tag_user_ids: Optional[list[str | int]] = None,
        used_in_card: bool = False,
        backticks: bool = False,
    ):
        message += f"\n{footer}"
        if not tag_user_ids:
            tag_user_ids = []
        elif isinstance(tag_user_ids, (str, int)):
            tag_user_ids = [tag_user_ids]
        elif not isinstance(tag_user_ids, list):
            raise Exception(f"Invalid {tag_user_ids=}")

        if tag_user_ids:
            message += "" if used_in_card else "\n"
            message += " ".join(
                self._generate_user_tag(user_id) for user_id in tag_user_ids if user_id
            )

        gchat_limit = 4096 - 50
        messages = []
        while len(message) > gchat_limit:
            part = message[: gchat_limit - 3] + "..."
            messages.append(part)
            message = message[gchat_limit - 3 :]
        messages.append(message)

        responses = []
        for message in messages:
            if message == messages[-1]:
                footer = ""
            data = json.dumps({"text": f"```{message}```" if backticks else message})
            headers = {"Content-Type": "application/json; charset=UTF-8"}
            response = requests.post(webhook, headers=headers, data=data, timeout=10)
            if response.status_code >= 300:
                raise Exception(
                    f"Error sending message. Status {response.status_code}: {response.text}"
                )
            responses.append(response.text)
        return "\n".join(responses)

    def send_card(
        self,
        sections: Union[Sections, Any],
        webhook: str,
        title: str,
        subtitle: str,
        footer: Optional[Union[Section, str]] = None,
        tag_user_ids: Optional[list] = None,
        card_id: Optional[str] = None,
    ):
        if sections.__class__.__name__ == "Sections" or isinstance(sections, Sections):
            sections_checked: Sections = sections
        elif isinstance(sections, list):
            sections_checked: Sections = Sections(*sections)
        elif sections.__class__.__name__ == "Section" or isinstance(sections, Section):
            sections_checked: Sections = Sections(sections)
        else:
            raise Exception(
                f"Func send_card: Not a Section/Sections/list\n"
                f"{type(sections)=}\n"
                f"{sections=}"
            )

        if footer:
            sections_checked.append(
                GoogleChat.text_to_section(footer)
                if isinstance(footer, str)
                else footer
            )

        now = datetime.now(pytz.timezone("Asia/Singapore"))

        json_payload = self._sections_to_payload(
            sections=sections_checked,
            title=title,
            card_id=card_id or "",
            subtitle=subtitle or f"{now:%Y-%m-%d %H:%M}",
        )
        response = requests.post(
            webhook,
            headers={
                "Content-Type": "application/json; charset=UTF-8",
            },
            data=json.dumps(json_payload),
            timeout=10,
        )
        if response.status_code >= 300:
            raise Exception(
                [
                    self.text_to_section_fancy(
                        header="⛔Error sending message",
                        top_label=f"Status {response.status_code}",
                        text=response.text,
                    )
                ]
            )
        if tag_user_ids:
            self.send_gchat(
                message=f"⬆{title}",
                webhook=webhook,
                tag_user_ids=tag_user_ids,
                footer="",
                used_in_card=True,
            )
        return response.text


def main():
    gchat = GoogleChat()
    sections = Sections(
        GoogleChat.text_to_section("Hello world"),
        GoogleChat.text_to_section("This is a test"),
    )
    gchat.send_card(
        sections=sections,
        webhook="https://chat.googleapis.com/v1/spaces/AAAAgmo6To4/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=-c-wZglE8nVvMJmlkoGp2s-Hxvd7VJ0nwQeHlTYXW3Y",
        title="Test",
        subtitle="Test",
        footer="Test",
        tag_user_ids=[1],
    )


if __name__ == "__main__":
    main()
