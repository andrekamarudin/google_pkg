import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from loguru import logger


@dataclass
class GService:
    service_key: Optional[dict] = field(default=None)
    service_key_path: Optional[Path | str] = field(default=None)
    service_key_env_var: Optional[str] = field(default=None)

    def __post_init__(self):
        assert any(
            [self.service_key, self.service_key_path, self.service_key_env_var]
        ), "service_key, service_key_path, or service_key_env_var must be provided."

        if self.service_key_env_var:
            logger.success(f"Using service_key_env_var: {self.service_key_env_var}")
            self.service_key_path = Path(
                os.environ[self.service_key_env_var],
            )

        if self.service_key_path:
            logger.success(f"Using service_key_path: {self.service_key_path}")
            if isinstance(self.service_key_path, str):
                self.service_key_path = Path(self.service_key_path)
            assert self.service_key_path.exists(), (
                f"service_key_path provided not found: {self.service_key_path}"
            )
            self.service_key = json.loads(self.service_key_path.read_text())

        if not self.service_key:
            raise Exception(
                "Unexpected types:\n" + f"service_key: {type(self.service_key)=}\n"
                if self.service_key
                else "" + f"service_key_path: {type(self.service_key_path)=}\n"
                if self.service_key_path
                else "" + f"service_key_env_var: {type(self.service_key_env_var)=}\n"
                if self.service_key_env_var
                else ""
            )

    def build_service(self, scopes, short_name, version, credentials=None):
        self.credentials = (
            credentials
            or service_account.Credentials.from_service_account_info(
                info=self.service_key, scopes=scopes
            )
        )
        self.service = build(short_name, version, credentials=self.credentials)
        logger.success("Initialized GService")
        return self.service


def main():
    g_service = GService(
        service_key_env_var="DBDA_GOOGLE_APPLICATION_CREDENTIALS",
    )
    logger.success("Service key loaded successfully.")


if __name__ == "__main__":
    logger.remove()
    LOG_FMT = "<level>{level}: {message}</level> <black>({file} / {module} / {function} / {line})</black>"
    logger.add(sys.stdout, level="SUCCESS", format=LOG_FMT)
    main()
