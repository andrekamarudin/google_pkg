# Gmail API Module (`gmail.py`)

This module provides a Python interface for interacting with the Gmail API. It includes classes for handling authentication, searching, reading, and managing emails.

## `EmailHeaders`

A Pydantic model for storing and validating email header information.

### Attributes

- `message_id` (Optional[str]): The message ID of the email.
- `thread_id` (Optional[str]): The thread ID of the email.
- `subject` (Optional[str]): The subject of the email.
- `to` (Optional[str]): The recipient(s) of the email.
- `cc` (Optional[str]): The CC recipient(s) of the email.
- `bcc` (Optional[str]): The BCC recipient(s) of the email.
- `date` (Optional[str]): The date the email was sent.
- `delivered_to` (Optional[str]): The recipient the email was delivered to.
- `sent_from` (Optional[str]): The sender of the email.
- `received` (Optional[str]): Information about how the email was received.
- `body` (Optional[str]): The body of the email.

## `GmailService`

The base class for interacting with the Gmail API. It handles authentication and provides low-level methods for accessing and modifying email data.

### Initialization

```python
__init__(self, credentials_path: Optional[Path] = None, token_path: Optional[Path] = None, *args, **kwargs)
```

- `credentials_path` (Optional[Path]): The path to the OAuth credentials file. Defaults to `credentials.json`.
- `token_path` (Optional[Path]): The path to the token file. Defaults to `token.pickle`.

### Methods

- `_get_oauth_credentials()`: Handles OAuth2 authentication and token management.
- `_get_msg_payload(msg_id: str) -> Optional[str]`: Retrieves the raw payload of a message.
- `_message_to_content(email_message: Message) -> str | None`: Extracts the text content from an email message.
- `_payload_to_body(message_payload: str) -> str`: Converts a raw message payload to a string body.
- `_payload_to_headers(message_payload: str) -> EmailHeaders`: Converts a raw message payload to an `EmailHeaders` object.
- `get_label_id_by_name(label_name: str) -> Optional[str]`: Retrieves the ID of a label by its name.
- `_mod_label(msg_id: str, label: Label, to_add: bool = True) -> Dict[str, Any] | None`: Adds or removes a label from a message.

## `GmailUI`

A subclass of `GmailService` that provides a command-line user interface for managing emails.

### Methods

- `search(query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]`: Searches for emails matching a given query.
- `move_to_inbox(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Moves an email to the inbox.
- `mark_as_ttd(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Marks an email with the 'TTD' (To-Do) label.
- `mark_as_read(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Marks an email as read.
- `delete_email(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Moves an email to the trash.
- `archive_email(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Archives an email.
- `mark_as_spam(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Marks an email as spam.
- `open_msg(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Opens an email in the browser.
- `msg_id_to_headers(msg_id: str) -> EmailHeaders`: Retrieves the headers for a given message ID.
- `handle_sender(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Handles all emails from the same sender.
- `handle_thread(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Handles all emails in the same thread.
- `handle_subject(msg_id: str, thread_id: str, headers: Optional[EmailHeaders] = None)`: Handles all emails with the same subject.
- `main_loop(query: Optional[str] = None)`: Starts the main interactive loop for managing emails.
- `send_email(recipients: list[str], subject: str = "", body: str = "")`: Sends an email.

### Properties

- `email_decisions`: A dictionary mapping single-character commands to email action methods.
- `search_decisions`: A dictionary mapping single-character commands to batch processing methods.
- `all_decisions`: A dictionary combining `email_decisions` and `search_decisions`.
- `labels`: A list of all available labels in the user's account.
