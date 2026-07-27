import importlib.util
import inspect
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Set
from typing import cast

if TYPE_CHECKING:
    from .model_adapter import ModelAdapter

logger = logging.getLogger("fastrag.loader")

HookCallable = Callable[..., Any]
MaybeHook = Optional[HookCallable]


class HookName(str, Enum):
    INIT = "init"
    LOAD_MODEL = "load_model"
    SCORE = "score"
    SCORE_UNSTRUCTURED = "score_unstructured"
    CHAT = "chat"
    GET_SUPPORTED_LLM_MODELS = "get_supported_llm_models"


@dataclass
class HookRegistry:
    init: MaybeHook = None
    load_model: MaybeHook = None
    score: MaybeHook = None
    score_unstructured: MaybeHook = None
    chat: MaybeHook = None
    get_supported_llm_models: MaybeHook = None

    @classmethod
    def from_module(cls, module: Any) -> "HookRegistry":
        hooks: Dict[str, MaybeHook] = {}
        for hook in HookName:
            attr = getattr(module, hook.value, None)
            if callable(attr):
                hooks[hook.value] = attr
                logger.info("Found hook: %s", hook.value)
            else:
                hooks[hook.value] = None
        return cls(**hooks)

    def available(self) -> Set[HookName]:
        return {hook for hook in HookName if self.get(hook) is not None}

    def has(self, hook: HookName) -> bool:
        return self.get(hook) is not None

    def get(self, hook: HookName) -> MaybeHook:
        return cast(MaybeHook, getattr(self, hook.value))

    def require(self, hook: HookName) -> HookCallable:
        fn = self.get(hook)
        if fn is None:
            raise NotImplementedError(f"{hook.value} hook is not implemented in custom.py")
        return fn


class HookLoader:
    def __init__(self, code_dir: str, max_workers: int = 1):
        self.code_dir = os.path.abspath(code_dir)
        self.custom_module: Any = None
        self._module_name: Optional[str] = None
        self.max_workers = max_workers
        self.hooks = HookRegistry()

    async def load(self) -> "ModelAdapter":
        custom_path = self._validate_paths()
        self.custom_module = self._import_custom_module(custom_path)
        self.hooks = HookRegistry.from_module(self.custom_module)
        self._validate_hooks_consistency()
        adapter = self._create_adapter(self.hooks)
        await adapter.initialize()
        return adapter

    def _validate_paths(self) -> str:
        if not os.path.exists(self.code_dir):
            raise FileNotFoundError(f"Code directory not found: {self.code_dir}")

        custom_path = os.path.join(self.code_dir, "custom.py")
        if not os.path.isfile(custom_path):
            raise FileNotFoundError(f"custom.py not found in code directory: {self.code_dir}")

        return custom_path

    def _import_custom_module(self, custom_path: str) -> Any:
        inserted_code_dir = False
        module_name = None

        try:
            if self._module_name is not None:
                sys.modules.pop(self._module_name, None)
                self._module_name = None

            if self.code_dir not in sys.path:
                sys.path.insert(0, self.code_dir)
                inserted_code_dir = True

            module_name = f"_fastrag_user_custom_module_{uuid.uuid4().hex}"
            self._module_name = module_name
            spec = importlib.util.spec_from_file_location(module_name, custom_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {custom_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            if module_name is not None:
                sys.modules.pop(module_name, None)
            self._module_name = None
            logger.exception("Failed to import custom.py from %s", self.code_dir)
            raise
        finally:
            if inserted_code_dir and self.code_dir in sys.path:
                sys.path.remove(self.code_dir)

    def _create_adapter(self, hooks: HookRegistry) -> "ModelAdapter":
        available_hooks = hooks.available()
        if not available_hooks:
            raise RuntimeError("No hooks avaiable in custom.py")

        from .model_adapter import AsyncModelAdapter  # noqa: PLC0415
        from .model_adapter import SyncModelAdapter  # noqa: PLC0415

        if any(self._is_async_hook(hooks.require(hook_name)) for hook_name in available_hooks):
            return AsyncModelAdapter(hooks=hooks, code_dir=self.code_dir)

        return SyncModelAdapter(hooks=hooks, code_dir=self.code_dir, max_workers=self.max_workers)

    def _is_async_hook(self, hook: HookCallable) -> bool:
        return inspect.iscoroutinefunction(hook) or (
            not inspect.isfunction(hook)
            and hasattr(hook, "__call__")
            and inspect.iscoroutinefunction(hook.__call__)
        )

    def _validate_hooks_consistency(self) -> None:
        available_hooks = self.hooks.available()
        async_hooks = []
        sync_hooks = []

        for hook_name in available_hooks:
            hook = self.hooks.get(hook_name)
            if hook and self._is_async_hook(hook):
                async_hooks.append(hook_name)
            else:
                sync_hooks.append(hook_name)

        if async_hooks and sync_hooks:
            raise ValueError(
                f"Hooks must be either all synchronous or all asynchronous. "
                f"Found async hooks: {[h.value for h in async_hooks]} and "
                f"sync hooks: {[h.value for h in sync_hooks]}."
            )
