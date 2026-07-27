import asyncio
import importlib
import inspect
import logging
import os
import threading
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pandas as pd
from opentelemetry import context

from .loader import HookCallable
from .loader import HookName
from .loader import HookRegistry

logger = logging.getLogger("fastrag.model_adapter")

_thread_local = threading.local()

_MODERATION_CONFIG_FILE = "moderation_config.yaml"
# datarobot_dome.drum_integration was deprecated in favour of datarobot_moderation_interface
_MODERATION_MODULES = [
    "datarobot_moderation_interface.drum_integration",
    "datarobot_dome.drum_integration",
]


class ModelAdapter(ABC):
    def __init__(self, hooks: HookRegistry, code_dir: str):
        self.hooks = hooks
        self.code_dir = code_dir
        self.model: Any = None
        self._mod_pipeline: Optional[Any] = None

    def _load_moderation_pipeline(self) -> Optional[Any]:
        """Load a datarobot_dome moderation pipeline if moderation_config.yaml is present."""
        config_path = os.path.join(self.code_dir, _MODERATION_CONFIG_FILE)
        if not os.path.exists(config_path):
            return None
        moderation_pipeline_factory = None
        for module_name in _MODERATION_MODULES:
            try:
                mod = importlib.import_module(module_name)
                moderation_pipeline_factory = mod.moderation_pipeline_factory
                break
            except ImportError:
                continue
        if moderation_pipeline_factory is None:
            logger.warning(
                "datarobot_dome not installed; moderation_config.yaml found "
                "but moderation is disabled"
            )
            return None
        target_type = self._resolve_target_type()
        pipeline = moderation_pipeline_factory(target_type, model_dir=self.code_dir)
        if pipeline is not None:
            logger.info(
                "Moderation pipeline loaded: %s (target_type=%s)",
                type(pipeline).__name__,
                target_type,
            )
        else:
            logger.info("moderation_pipeline_factory returned None for target_type=%s", target_type)
        return pipeline

    def _resolve_target_type(self) -> str:
        """Return target type string for moderation factory.

        Prefers TARGET_TYPE env var, then model-metadata.yaml, then falls back to "regression".
        """
        if env_tt := os.environ.get("TARGET_TYPE"):
            return env_tt.lower()
        metadata_path = os.path.join(self.code_dir, "model-metadata.yaml")
        if os.path.exists(metadata_path):
            try:
                import yaml  # noqa: PLC0415

                with open(metadata_path) as f:
                    data = yaml.safe_load(f) or {}
                return str(data.get("targetType", "regression")).lower()
            except Exception as exc:
                logger.warning("Could not read model-metadata.yaml for target_type: %s", exc)
        return "regression"

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def score(self, data: pd.DataFrame, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def chat(self, completion_create_params: Dict[str, Any], **kwargs: Any) -> Any: ...

    @abstractmethod
    async def score_unstructured(self, data: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def get_supported_llm_models(self) -> Any: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class AsyncModelAdapter(ModelAdapter):
    """All hooks are native coroutines -- awaits them directly."""

    async def initialize(self) -> None:
        if self.hooks.has(HookName.INIT):
            await self.hooks.require(HookName.INIT)(code_dir=self.code_dir)

        if self.hooks.has(HookName.LOAD_MODEL):
            self.model = await self.hooks.require(HookName.LOAD_MODEL)(code_dir=self.code_dir)
        else:
            logger.warning("No load_model hook found. Model will be None.")

        self._mod_pipeline = self._load_moderation_pipeline()

    async def score(self, data: pd.DataFrame, **kwargs: Any) -> Any:
        score_hook = self.hooks.require(HookName.SCORE)
        if self._mod_pipeline is not None:
            return await self._mod_pipeline.async_score(data, self.model, score_hook, **kwargs)
        return await score_hook(data, self.model, **kwargs)

    async def chat(self, completion_create_params: Dict[str, Any], **kwargs: Any) -> Any:
        chat_hook = self.hooks.require(HookName.CHAT)
        if self._mod_pipeline is not None:
            return await self._mod_pipeline.async_chat(
                completion_create_params, self.model, chat_hook, **kwargs
            )
        return await chat_hook(completion_create_params, self.model, **kwargs)

    async def score_unstructured(self, data: Any, **kwargs: Any) -> Any:
        return await self.hooks.require(HookName.SCORE_UNSTRUCTURED)(data, self.model, **kwargs)

    async def get_supported_llm_models(self) -> Any:
        return await self.hooks.require(HookName.GET_SUPPORTED_LLM_MODELS)(self.model)

    def shutdown(self) -> None:
        pass


class SyncModelAdapter(ModelAdapter):
    """Sync hooks run in a ThreadPoolExecutor with OTel context propagation."""

    def __init__(self, hooks: HookRegistry, code_dir: str, max_workers: int = 1):
        super().__init__(hooks, code_dir)
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_loops: list[asyncio.AbstractEventLoop] = []
        self._worker_loops_lock = threading.Lock()

    async def initialize(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            initializer=self._make_thread_initializer(),
        )

        logger.info("Warming up worker threads...")
        warmup_futures = [self._executor.submit(lambda: None) for _ in range(self.max_workers)]
        for future in warmup_futures:
            future.result()

        self._mod_pipeline = self._load_moderation_pipeline()

        if self.hooks.has(HookName.LOAD_MODEL):
            if self._mod_pipeline is not None:
                # Load a shared model instance for the async moderation path.
                # The async pipeline methods (async_chat/async_score) take model as an explicit
                # parameter — they don't use thread-local storage — so we need self.model set.
                # Run in the executor so load_model executes in an already-initialised worker
                # thread (thread-local event loop present for libs like NeMo GuardRails).
                load_hook = self.hooks.require(HookName.LOAD_MODEL)
                code_dir = self.code_dir
                loop = asyncio.get_running_loop()
                self.model = await loop.run_in_executor(
                    self._executor, lambda: load_hook(code_dir=code_dir)
                )
            else:
                logger.info("load_model hook is sync; executed eagerly in worker threads.")
        else:
            logger.warning("No load_model hook found. Model will be None.")

    def _make_thread_initializer(self) -> Callable[[], None]:
        init_hook = self.hooks.get(HookName.INIT)
        load_model_hook = self.hooks.get(HookName.LOAD_MODEL)
        code_dir = self.code_dir
        worker_loops = self._worker_loops
        worker_loops_lock = self._worker_loops_lock

        def thread_initializer() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with worker_loops_lock:
                worker_loops.append(loop)
            if init_hook:
                init_hook(code_dir=code_dir)
            if load_model_hook:
                try:
                    _thread_local.model = load_model_hook(code_dir=code_dir)
                except Exception as exc:
                    logger.error("Failed to run thread-local load_model: %s", exc)
                    raise

        return thread_initializer

    async def _run_in_executor(self, hook: HookCallable, *args: Any, **kwargs: Any) -> Any:
        ctx = context.get_current()

        def _run() -> Any:
            token = context.attach(ctx)
            try:
                model = getattr(_thread_local, "model", self.model)
                return hook(*args, model, **kwargs)
            finally:
                context.detach(token)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _run)

        if inspect.isawaitable(result):
            return await result
        return result

    async def score(self, data: pd.DataFrame, **kwargs: Any) -> Any:
        score_hook = self.hooks.require(HookName.SCORE)
        if self._mod_pipeline is not None:
            return await self._mod_pipeline.async_score(data, self.model, score_hook, **kwargs)
        return await self._run_in_executor(score_hook, data, **kwargs)

    async def chat(self, completion_create_params: Dict[str, Any], **kwargs: Any) -> Any:
        chat_hook = self.hooks.require(HookName.CHAT)
        if self._mod_pipeline is not None:
            return await self._mod_pipeline.async_chat(
                completion_create_params, self.model, chat_hook, **kwargs
            )
        return await self._run_in_executor(chat_hook, completion_create_params, **kwargs)

    async def score_unstructured(self, data: Any, **kwargs: Any) -> Any:
        return await self._run_in_executor(
            self.hooks.require(HookName.SCORE_UNSTRUCTURED), data, **kwargs
        )

    async def get_supported_llm_models(self) -> Any:
        return await self._run_in_executor(self.hooks.require(HookName.GET_SUPPORTED_LLM_MODELS))

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            for loop in self._worker_loops:
                if not loop.is_closed():
                    loop.close()
            self._worker_loops.clear()
            self._executor = None
