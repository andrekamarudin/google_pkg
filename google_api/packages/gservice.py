import json
import os
import sys
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from loguru import logger
from pydantic import BaseModel


class ServiceKey(BaseModel):
    type: str
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    universe_domain: str


class GService:
    """
    A class representing a Google Service.
    """

    def __init__(
        self,
        service_key: Optional[ServiceKey | dict] = None,
        service_key_path: Optional[Path | str] = None,
    ):
        logger.info("Initializing GService")

        if isinstance(service_key_path, str | Path):
            self.service_key_path: Path = (
                Path(service_key_path)
                if isinstance(service_key_path, str)
                else service_key_path
            )
        else:
            self.service_key_path = None

        if isinstance(service_key, ServiceKey):
            service_key_checked: ServiceKey = service_key
        elif isinstance(service_key, dict):
            service_key_checked: ServiceKey = ServiceKey(**service_key)
        elif isinstance(service_key_path, str) and Path(service_key_path).exists():
            service_key_checked: ServiceKey = ServiceKey(
                **json.loads(Path(service_key_path).read_text())
            )

        elif isinstance(service_key_path, Path) and service_key_path.exists():
            service_key_checked: ServiceKey = ServiceKey(
                **json.loads(service_key_path.read_text())
            )
            self.service_key_path = service_key_path
        elif GOOGLE_APPLICATION_CREDENTIALS := os.environ.get(
            "DBDA_GOOGLE_APPLICATION_CREDENTIALS"
        ) or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            self.service_key_path: Path = Path(GOOGLE_APPLICATION_CREDENTIALS)
            if self.service_key_path.exists():
                service_key_checked: ServiceKey = ServiceKey(
                    **json.loads(self.service_key_path.read_text())
                )
            else:
                raise Exception(
                    f"Provided Service account JSON file path does not exist: {self.service_key_path}"
                )
        else:
            raise Exception(
                "Service account JSON file not found in system or environment variables"
            )

        if isinstance(service_key_checked, ServiceKey):
            self.sa_info: dict = service_key_checked.model_dump()
            logger.success("Initialized GService")

    def build_service(self, scopes, short_name, version, credentials=None):
        self.credentials = (
            credentials
            or service_account.Credentials.from_service_account_info(
                info=self.sa_info, scopes=scopes
            )
        )
        self.service = build(short_name, version, credentials=self.credentials)
        logger.success("Initialized GService")
        return self.service


def main():
    g_service = GService()
    print(type(g_service.sa_info))
    print((g_service.sa_info))


if __name__ == "__main__":
    logger.remove()
    LOG_FMT = "<level>{level}: {message}</level> <black>({file} / {module} / {function} / {line})</black>"
    logger.add(sys.stdout, level="SUCCESS", format=LOG_FMT)
    main()
