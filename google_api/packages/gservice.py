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
        service_key_env_var: Optional[str] = None,
    ):
        logger.info("Initializing GService")

        assert (
            service_key
            or service_key_path
            or service_key_env_var
            or os.getenv("DBDA_GOOGLE_APPLICATION_CREDENTIALS")
        ), (
            "service_key, service_key_path, or service_key_env_var must be provided\n"
            + f"service_key: {service_key=}\n"
            + f"service_key_path: {service_key_path=}\n"
            + f"service_key_env_var: {service_key_env_var=}\n"
            + f"DBDA_GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('DBDA_GOOGLE_APPLICATION_CREDENTIALS')=}\n"
        )

        if service_key and isinstance(service_key, ServiceKey):
            self.service_key = service_key
        elif service_key_env_var:
            service_key_path_str: str = os.environ[service_key_env_var]
            self.service_key_path: Path = Path(service_key_path_str)
            assert (
                self.service_key_path.exists()
            ), f"service_key_env_var provided not found: {service_key_path_str}"
            self.service_key: ServiceKey = ServiceKey(
                **json.loads(self.service_key_path.read_text())
            )
        elif isinstance(service_key_path, str | Path):
            service_key_path: Path = (
                Path(service_key_path)
                if isinstance(service_key_path, str)
                else service_key_path
            )
            self.service_key_path = service_key_path
            assert (
                self.service_key_path.exists()
            ), f"service_key_path provided not found: {service_key_path}"
            service_key_path_str: str = self.service_key_path.read_text()
            self.service_key: ServiceKey = ServiceKey(
                **json.loads(service_key_path_str)
            )
        elif GOOGLE_APPLICATION_CREDENTIALS := os.environ.get(
            "DBDA_GOOGLE_APPLICATION_CREDENTIALS"
        ):
            self.service_key_path: Path = Path(GOOGLE_APPLICATION_CREDENTIALS)
            assert self.service_key_path.exists(), f"DBDA_GOOGLE_APPLICATION_CREDENTIALS provided not found: {GOOGLE_APPLICATION_CREDENTIALS}"
            service_key_path_str: str = self.service_key_path.read_text()
            self.service_key: ServiceKey = ServiceKey(
                **json.loads(service_key_path_str)
            )
        else:
            raise Exception(
                "Unexpected types:\n" + f"service_key: {type(service_key)=}\n"
                if service_key
                else "" + f"service_key_path: {type(service_key_path)=}\n"
                if service_key_path
                else "" + f"service_key_env_var: {type(service_key_env_var)=}\n"
                if service_key_env_var
                else ""
            )

        self.sa_info: dict = self.service_key.model_dump()

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
