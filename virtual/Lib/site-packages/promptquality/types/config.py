from getpass import getpass
from pathlib import Path
from posixpath import join
from time import time
from typing import Optional
from urllib.parse import urljoin
from webbrowser import open_new_tab

from jwt import decode
from pydantic import (
    UUID4,
    HttpUrl,
    SecretStr,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from promptquality.constants.prompt_optimization import PromptOptimizationDefaults
from promptquality.constants.routes import Routes
from promptquality.constants.run import RunDefaults
from promptquality.types.run import (
    BaseTemplateResponse,
    CreateTemplateVersionResponse,
    Dataset,
    JobInfoMixin,
    ProjectResponse,
    RunResponse,
)
from promptquality.utils.api_client import ApiClient
from promptquality.utils.config import get_config_location
from promptquality.utils.logger import logger


class Config(BaseSettings):
    console_url: HttpUrl
    # User.
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    api_key: Optional[SecretStr] = None
    token: Optional[SecretStr] = None
    current_user: Optional[str] = None
    # Project.
    current_project_id: Optional[UUID4] = None
    current_project_name: Optional[str] = None
    # Run.
    current_run_id: Optional[UUID4] = None
    current_run_name: Optional[str] = None
    current_run_url: Optional[str] = None
    current_run_task_type: Optional[int] = None
    # Template.
    current_template_id: Optional[UUID4] = None
    current_template_name: Optional[str] = None
    # Version
    current_template_version_id: Optional[UUID4] = None
    current_template_version: Optional[int] = None
    current_template: Optional[str] = None
    # Dataset.
    current_dataset_id: Optional[UUID4] = None
    # Job.
    current_job_id: Optional[UUID4] = None
    current_prompt_optimization_job_id: Optional[UUID4] = None

    model_config = SettingsConfigDict(
        # Allow loading from environment variables.
        env_prefix="GALILEO_",
        # Ignore unknown fields when loading from a config file.
        extra="ignore",
        # Validate fields at assignment.
        validate_assignment=True,
    )

    @computed_field
    def api_url(self) -> HttpUrl:
        # Local dev.
        if self.console_url.host in ["localhost", "127.0.0.1"]:
            api_url = "http://localhost:8088"
        else:
            api_url = self.console_url.unicode_string().replace("console", "api")
        return HttpUrl(api_url)

    @property
    def config_file(self) -> Path:
        return get_config_location()

    @field_validator("console_url", mode="before")
    def http_url(cls, value: str) -> str:
        if value and not (value.startswith("https") or value.startswith("http")):
            value = f"https://{value}"
        return value

    @model_validator(mode="after")
    def validate_api_url(self) -> "Config":
        if not ApiClient.healthcheck(str(self.api_url)):
            # TODO: Make this a custom error.
            raise ValidationError(f"Could not connect to {self.api_url}.")
        return self

    @field_serializer("token", when_used="json")
    def serialize_token(self, value: SecretStr) -> str:
        return value.get_secret_value()

    @classmethod
    def read(cls) -> "Config":
        return cls.model_validate_json(get_config_location().read_text())

    def write(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(self.model_dump_json(exclude_none=True))

    @property
    def project_url(self) -> str:
        if not self.current_project_id:
            raise ValueError("No project set.")
        return join(
            self.console_url.unicode_string(),
            "prompt",
            "chains" if self.current_run_task_type == RunDefaults.prompt_chain_task_type else "",
            f"{self.current_project_id}",
        )

    @property
    def api_client(self) -> ApiClient:
        self._refresh_token()
        if not self.token:
            raise ValueError("No token set. Please log in.")
        return ApiClient(api_url=self.api_url, token=self.token)

    def token_login(self) -> str:
        token_url = urljoin(str(self.console_url), Routes.get_token)
        logger.info(f"🔐 Opening {token_url} to generate a new Galileo token.")
        try:
            open_new_tab(token_url)
        except Exception:
            pass
        finally:
            print(f"Go to {token_url} to generate a new Galileo token.")
            return getpass("🔐 Enter your Galileo token:")

    def login(self) -> None:
        if self.api_key:
            token = ApiClient.api_key_login(str(self.api_url), self.api_key.get_secret_value()).get("access_token", "")
        elif self.username and self.password:
            token = ApiClient.username_login(str(self.api_url), self.username, self.password.get_secret_value()).get(
                "access_token", ""
            )
        else:
            token = self.token_login()
        self.token = SecretStr(token)
        self.current_user = self.api_client.get_current_user().get("email")
        self.write()
        print(f"👋 You have logged into 🔭 Galileo ({self.console_url}) as {self.current_user}.")

    def _refresh_token(self) -> None:
        """Refresh token if not present or expired."""
        # Check to see if our token is expired before making a request
        # and refresh token if it's expired
        if self.token:
            claims = decode(self.token.get_secret_value(), options={"verify_signature": False})
            if claims.get("exp", 0) < time():
                self.login()

    def logout(self) -> None:
        # Reset credentials.
        self.username = None
        self.password = None
        self.token = None
        self.current_user = None
        # Reset all other values.
        self.current_project_id = None
        self.current_project_name = None
        self.current_run_id = None
        self.current_run_name = None
        self.current_run_task_type = None
        self.current_template_id = None
        self.current_template_name = None
        self.current_template_version_id = None
        self.current_template_version = None
        self.current_template = None
        self.current_dataset_id = None
        self.current_job_id = None
        self.current_prompt_optimization_job_id = None
        self.write()
        print("👋 You have logged out of 🔭 Galileo.")

    def merge_project(self, project: ProjectResponse) -> None:
        self.current_project_id = project.id
        self.current_project_name = project.name
        self.write()
        logger.debug(f"📝 Set current project to {project.name}.")

    def merge_template(self, template: BaseTemplateResponse) -> None:
        self.current_template_id = template.id
        self.current_template_name = template.name
        self.merge_template_version(template.selected_version)
        self.write()
        logger.debug(f"📝 Set current template to {template.name}.")

    def merge_template_version(self, template_version: CreateTemplateVersionResponse) -> None:
        self.current_template_version_id = template_version.id
        self.current_template_version = template_version.version
        self.current_template = template_version.template
        self.write()
        logger.debug(f"📝 Set current template version to {template_version.version}.")

    def merge_dataset(self, dataset: Dataset) -> None:
        self.current_dataset_id = dataset.id
        self.write()
        logger.debug(f"📝 Set current dataset to {dataset.id}.")

    def merge_run(self, run: RunResponse) -> None:
        self.current_run_id = run.id
        self.current_run_name = run.name
        self.current_run_task_type = run.task_type
        self.write()
        logger.debug(f"📝 Set current run to {run.name}.")

    def merge_job(self, job_info: JobInfoMixin) -> None:
        self.current_job_id = job_info.job_id
        self.current_run_url = job_info.link
        if job_info.job_name == PromptOptimizationDefaults.prompt_optimization_job_name:
            self.current_prompt_optimization_job_id = job_info.job_id
        self.write()
        logger.debug(f"📝 Set current job to {job_info.job_id}.")
