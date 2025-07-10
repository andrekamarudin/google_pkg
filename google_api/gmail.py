# %%
import asyncio
import base64
import os
import pickle
from email import policy
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.parser import BytesParser
from email.utils import formatdate
from enum import Enum
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Callable, Dict, List, Optional

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from loguru import logger
from packages.gservice import GService
from packages.helper import condense_text
from pydantic import BaseModel
from tqdm import tqdm


class Criteria(Enum):
    SENDER = {"operator": "from", "property": "sent_from"}
    SUBJECT = {"operator": "subject", "property": "subject"}
    THREAD_ID = {"operator": "thread_id", "property": "thread_id"}


class EmailHeaders(BaseModel):
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    subject: Optional[str] = None
    to: Optional[str] = None
    cc: Optional[str] = None
    bcc: Optional[str] = None
    date: Optional[str] = None
    delivered_to: Optional[str] = None
    sent_from: Optional[str] = None
    received: Optional[str] = None
    body: Optional[str] = None

    def __init__(self, **kwargs) -> None:
        kwargs = {k.lower().replace("-", "_"): v for k, v in kwargs.items()}
        kwargs["sent_from"] = kwargs.pop("from")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        repr_dict = self.__dict__.copy()
        repr_dict["body"] = condense_text(repr_dict["body"])
        return f"{self.__class__.__name__}({pformat(repr_dict)})"


class Label(Enum):
    INBOX = "INBOX"
    SPAM = "SPAM"
    TRASH = "TRASH"
    UNREAD = "UNREAD"
    TTD = "Label_1833152009763122946"


class GmailService:
    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> None:
        self.SCOPES: List[str] = [
            f"https://www.googleapis.com/auth/gmail.{action}"
            for action in ["readonly", "modify", "send"]
        ]

        # Set default paths
        self.credentials_path = credentials_path or Path("credentials.json")
        self.token_path = token_path or Path("token.pickle")

        self.credentials = self._get_oauth_credentials()
        self.cred_file_folder: Path = Path(os.environ["USERPROFILE"])
        self.user_id: str = "me"

        self.gservice = GService(service_key_path=self.credentials_path)
        self.gservice.build_service(
            scopes=self.SCOPES,
            short_name="gmail",
            version="v1",
            credentials=self.credentials,
        )

    def _get_oauth_credentials(self):
        """Get OAuth2 credentials using the installed app flow"""
        creds = None

        # Check if token.pickle exists (saved credentials)
        if self.token_path.exists():
            with open(self.token_path, "rb") as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"OAuth credentials file not found at {self.credentials_path}. "
                        "Download it from Google Cloud Console."
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_path, "wb") as token:
                pickle.dump(creds, token)

        return creds

    def _get_msg_payload(self, msg_id: str) -> Optional[str]:
        try:
            message = (
                self.gservice.service.users()
                .messages()
                .get(userId=self.user_id, id=msg_id, format="raw")
                .execute()
            )
        except Exception as e:
            logger.warning(f"Error getting message {msg_id}: {e}. Skipping...")
            return None
        if not (message_payload := message.get("raw")):
            logger.info(f"No payload found for message {msg_id}")
            logger.info(f"message: {message}")
            return None
        return message_payload

    def _message_to_content(self, email_message: Message) -> str | None:
        assert isinstance(email_message, Message), (
            f"email_message ({type(email_message)}): {str(email_message)}"
        )
        content_type: str = email_message.get_content_type()
        if content_type == "text/plain":
            return email_message.get_content()
        elif content_type == "text/html":
            html_content = email_message.get_content()
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text()
        elif content_type == "multipart/alternative":
            for part in email_message.iter_parts():
                content = self._message_to_content(part)
            if content:
                return content
            else:
                return "get_content: (no content found in multipart/alternative])"
        elif content_type == "application/octet-stream":
            content = "(application/octet-stream content found)"
            return content

    def _payload_to_body(self, message_payload: str) -> str:
        raw_email: bytes = base64.urlsafe_b64decode(message_payload)
        email_message: Message = BytesParser(policy=policy.default).parsebytes(
            raw_email
        )
        body: str = self._message_to_content(email_message) or ""
        if not body and email_message.is_multipart():
            body = ""
            for part in email_message.iter_parts():
                body += self._message_to_content(part) or ""
        if body:
            return body
        else:
            logger.error(f"Failed to parse body for {message_payload[:100]}...")
            return "(no body found)"

    def _payload_to_headers(self, message_payload: str) -> EmailHeaders:
        email_bytes: bytes = base64.urlsafe_b64decode(message_payload)
        message: Message = BytesParser(policy=policy.default).parsebytes(email_bytes)
        email_dict: Dict[str, Any] = dict(message.items())
        email_dict["body"] = self._payload_to_body(message_payload)
        headers: EmailHeaders = EmailHeaders(**email_dict)
        return headers

    def get_label_id_by_name(self, label_name: str) -> Optional[str]:
        response: Dict[str, Any] = (
            self.gservice.service.users().labels().list(userId=self.user_id).execute()
        )
        for label in response["labels"]:
            if label["name"] == label_name:
                return label["id"]
        return None

    async def _mod_label(
        self, msg_id: str, label: Label, to_add: bool = True
    ) -> Dict[str, Any] | None:
        if label_id := label.value:
            response = await (
                self.gservice.service.users()
                .messages()
                .modify(
                    userId=self.user_id,
                    id=msg_id,
                    body={f"{'add' if to_add else 'remove'}LabelIds": [label_id]},
                )
                .execute()  # Remove await here
            )
            return response  # Return the response directly
        else:
            return None


class GmailUI(GmailService):
    def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        self.last_query = query
        msg_dicts: List[Dict[str, Any]] = (
            self.gservice.service.users()
            .messages()
            .list(userId=self.user_id, q=query)
            .execute()
        ).get("messages")
        if not msg_dicts:
            logger.warning("No messages found")
            return []
        else:
            logger.success("Found {} emails", len(msg_dicts))
            if limit:
                msg_dicts = msg_dicts[-limit:]
                logger.success("Returning last {} messages", limit)
            return msg_dicts

    async def move_to_inbox(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await asyncio.gather(
            self._mod_label(msg_id, Label.SPAM, to_add=False),
            self._mod_label(msg_id, Label.TRASH, to_add=False),
            self._mod_label(msg_id, Label.INBOX),
        )
        print(f"Moved {msg_id} to inbox email")

    async def mark_as_ttd(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await asyncio.gather(
            self._mod_label(msg_id, Label.TTD),
            self._mod_label(msg_id, Label.INBOX, to_add=False),
        )
        await self._mod_label(msg_id, Label.INBOX, to_add=False)
        print(f"Marked {msg_id} as TTD email")

    async def mark_as_read(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await asyncio.gather(
            self._mod_label(msg_id, Label.UNREAD, to_add=False),
        )
        print(f"Marked {msg_id} as read email")

    async def delete_email(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        try:
            await asyncio.gather(
                self._mod_label(msg_id, Label.TTD, to_add=False),
                self._mod_label(msg_id, Label.TRASH),
            )
            print(f"Deleted {msg_id}")
        except Exception as e:
            logger.error(f"Failed to delete email {msg_id}: {e}")

    async def archive_email(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await asyncio.gather(
            self._mod_label(msg_id, Label.INBOX, to_add=False),
            self._mod_label(msg_id, Label.TTD, to_add=False),
        )
        print(f"Archived {msg_id}")

    async def mark_as_spam(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await asyncio.gather(
            self._mod_label(msg_id, Label.SPAM),
            self._mod_label(msg_id, Label.INBOX, to_add=False),
        )
        print(f"Marked {msg_id} as spam")

    def open_msg(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        url = f"https://mail.google.com/mail/u/0/#inbox/{msg_id}"
        os.system(f'start "" "{url}"')

    def msg_id_to_headers(self, msg_id: str) -> EmailHeaders:
        payload = self._get_msg_payload(msg_id)
        return self._payload_to_headers(payload or "")

    def _show_options(self, options_dict: Dict[str, Callable]) -> str:
        return "\n".join(
            f"{key}: {func.__name__}" for key, func in options_dict.items()
        )

    async def _batch_process(
        self,
        msgs: List[Dict[str, Any]],
        decision: str,
        headers: Optional[EmailHeaders] = None,
    ) -> None:
        assert decision in self.email_decisions, f"Invalid decision: {decision}"
        function_to_call: Callable = self.email_decisions[decision]
        qbar = tqdm(
            total=len(msgs),
            desc=f"{decision} {len(msgs)} emails",
            unit="email",
            disable=len(msgs) < 10,  # Disable progress bar for small batches
        )

        async def act(
            msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
        ) -> None:
            try:
                function_to_call(msg_id, thread_id, headers)
            except Exception as e:
                logger.error(f"Error processing {msg_id}: {e}")
            qbar.update(1)

        await asyncio.gather(
            *[
                act(msg_id=msg["id"], thread_id=msg["threadId"], headers=headers)
                for msg in msgs
            ]
        )

    async def _batch_decision(
        self, criteria: Criteria, headers: Optional[EmailHeaders] = None
    ) -> None:
        operator = criteria.value["operator"]
        property = criteria.value["property"]
        keyword = getattr(headers, property)
        all_msgs = self.search(f"{operator}:{keyword}")
        print(f"Found {len(all_msgs)} emails with {operator} {keyword}")
        decision = ""
        while decision not in self.email_decisions:
            decision = input(self._show_options(self.email_decisions)).lower()
            if decision not in self.email_decisions:
                print("Invalid decision. Please try again.")
            else:
                break

        return await self._batch_process(all_msgs, decision, headers)

    async def handle_sender(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await self._batch_decision(Criteria.SENDER, headers)

    async def handle_thread(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await self._batch_decision(Criteria.THREAD_ID, headers)

    async def handle_subject(
        self, msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None
    ) -> None:
        await self._batch_decision(Criteria.SUBJECT, headers)

    @property
    def email_decisions(self) -> Dict[str, Callable]:
        return {
            "a": self.archive_email,
            "d": self.delete_email,
            "e": self.mark_as_ttd,
            "r": self.mark_as_read,
            "s": self.mark_as_spam,
            "o": self.open_msg,
        }

    @property
    def search_decisions(self) -> Dict[str, Callable]:
        return {
            "f": self.handle_sender,
            "t": self.handle_thread,
            "b": self.handle_subject,
        }

    @property
    def all_decisions(self) -> Dict[str, Callable]:
        return {**self.search_decisions, **self.email_decisions}

    def main_loop(self, query: Optional[str] = None) -> None:
        while True:
            search_query: str = query or input("Enter search query: ") or "is:unread"
            search_results: List[Dict[str, Any]] = self.search(search_query)
            for email in search_results:
                headers: EmailHeaders = self.msg_id_to_headers(email["id"])
                pprint(headers)
                decision: str = ""
                while decision not in self.all_decisions and decision != "q":
                    decision: str = input(
                        self._show_options(self.all_decisions)
                    ).lower()
                if decision == "q":
                    break
                self.all_decisions[decision](
                    msg_id=email["id"], thread_id=email["threadId"], headers=headers
                )

    def send_email(
        self,
        recipients: list[str],
        subject: str = "",
        body: str = "",
    ):
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.user_id
        message["To"] = ",".join(recipients)
        message["Date"] = formatdate(localtime=True)

        message.attach(MIMEText(body, "plain"))

        raw_message: str = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode()  # Encode to base64
        self.gservice.service.users().messages().send(
            userId=self.user_id,
            body={"raw": raw_message},  # Use the encoded message as the body
        ).execute()

    @property
    def labels(self) -> List[Dict[str, Any]]:
        response: dict = (
            self.gservice.service.users().labels().list(userId="me").execute()
        )
        return response.get("labels", [])


# %%
if __name__ == "__main__":
    gmail = GmailUI(
        credentials_path=Path(
            r"c:\Users\andre\OneDrive\Apps\OAUTH_GOOGLE_APPLICATION_CREDENTIALS.json"
        ),
        token_path=Path(r"c:\Users\andre\OneDrive\Apps\gmail_token.pickle"),
        service_key_path=Path(
            r"c:\Users\andre\OneDrive\Apps\ANDRE_GOOGLE_APPLICATION_CREDENTIALS.json"
        ),
    )
    gmail.main_loop()


# %%
