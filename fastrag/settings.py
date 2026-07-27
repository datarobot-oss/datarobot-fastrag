import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    code_dir: str = Field(default=".", validation_alias="CODE_DIR")
    address: str = Field(default="0.0.0.0:8080", validation_alias="ADDRESS")
    workers: int = Field(default=1, validation_alias="MAX_WORKERS")
    thread_pool_workers: int = Field(default=2, validation_alias="THREAD_POOL_WORKERS")
    runtime_params_file: Optional[str] = Field(default=None, validation_alias="RUNTIME_PARAMS_FILE")
    verbose: bool = Field(default=False, validation_alias="VERBOSE")
    allow_dr_api_access: bool = Field(
        default=False, validation_alias="ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"
    )
    model_config = SettingsConfigDict(extra="ignore")

    def setup_env(self) -> None:
        os.environ["CODE_DIR"] = os.path.abspath(self.code_dir)
        os.environ["ADDRESS"] = self.address
        os.environ["MAX_WORKERS"] = str(self.workers)
        os.environ["THREAD_POOL_WORKERS"] = str(self.thread_pool_workers)
        if self.runtime_params_file:
            os.environ["RUNTIME_PARAMS_FILE"] = os.path.abspath(self.runtime_params_file)
        os.environ["VERBOSE"] = "true" if self.verbose else "false"
        os.environ["ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"] = (
            "true" if self.allow_dr_api_access else "false"
        )

    def _merge_env_overrides(self) -> None:
        if code_dir := os.environ.get("CODE_DIR"):
            self.code_dir = code_dir
        if address := os.environ.get("ADDRESS"):
            self.address = address
        if workers := os.environ.get("MAX_WORKERS"):
            self.workers = int(workers)
        if thread_pool_workers := os.environ.get("THREAD_POOL_WORKERS"):
            self.thread_pool_workers = int(thread_pool_workers)
        if runtime_params_file := os.environ.get("RUNTIME_PARAMS_FILE"):
            self.runtime_params_file = runtime_params_file
        if verbose := os.environ.get("VERBOSE"):
            self.verbose = verbose.lower() in {"1", "true", "yes", "on"}
        if allow_dr_api_access := os.environ.get("ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"):
            self.allow_dr_api_access = allow_dr_api_access.lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    settings = Settings()
    settings._merge_env_overrides()
    return settings
