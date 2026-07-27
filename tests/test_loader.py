import sys

import pytest

from fastrag.loader import HookLoader
from fastrag.loader import HookName
from fastrag.loader import HookRegistry
from fastrag.model_adapter import AsyncModelAdapter
from fastrag.model_adapter import SyncModelAdapter


def test_hook_registry_from_module():
    class MockModule:
        def init(self, x):
            pass

        def score(self, x, y):
            pass

    mock_module = MockModule()
    registry = HookRegistry.from_module(mock_module)

    assert registry.init is not None
    assert registry.score is not None
    assert registry.load_model is None
    assert registry.chat is None

    assert registry.has(HookName.INIT) is True
    assert registry.has(HookName.SCORE) is True
    assert registry.has(HookName.LOAD_MODEL) is False

    assert registry.available() == {HookName.INIT, HookName.SCORE}


def test_hook_registry_require():
    registry = HookRegistry(init=lambda x: "init")
    assert registry.require(HookName.INIT)(None) == "init"

    with pytest.raises(NotImplementedError):
        registry.require(HookName.SCORE)


@pytest.mark.asyncio
async def test_hook_loader_load_missing_dir(tmp_path):
    loader = HookLoader(str(tmp_path / "non_existent"))
    with pytest.raises(FileNotFoundError, match="Code directory not found"):
        await loader.load()


@pytest.mark.asyncio
async def test_hook_loader_load_missing_custom_py(tmp_path):
    original_sys_path = list(sys.path)
    loader = HookLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="custom.py not found"):
        await loader.load()
    assert sys.path == original_sys_path


@pytest.mark.asyncio
async def test_hook_loader_load_success(tmp_path):
    code_dir = tmp_path / "model_dir"
    code_dir.mkdir()
    custom_py = code_dir / "custom.py"
    custom_py.write_text("""
async def init(code_dir):
    return "initialized"

async def load_model(code_dir):
    return "my_model"

async def score(data, model, **kwargs):
    return "scored"
""")

    loader = HookLoader(str(code_dir))
    adapter = await loader.load()

    assert str(code_dir) not in sys.path
    assert isinstance(adapter, AsyncModelAdapter)
    assert adapter.model == "my_model"
    assert loader.hooks.has(HookName.INIT)
    assert loader.hooks.has(HookName.LOAD_MODEL)
    assert loader.hooks.has(HookName.SCORE)
    assert not loader.hooks.has(HookName.CHAT)


@pytest.mark.asyncio
async def test_hook_loader_import_error(tmp_path):
    code_dir = tmp_path / "model_error"
    code_dir.mkdir()
    custom_py = code_dir / "custom.py"
    custom_py.write_text("import non_existent_module_foo_bar_baz")

    original_sys_path = list(sys.path)
    module_keys_before = {
        key for key in sys.modules if key.startswith("_fastrag_user_custom_module_")
    }

    loader = HookLoader(str(code_dir))
    with pytest.raises(ImportError):
        await loader.load()

    module_keys_after = {
        key for key in sys.modules if key.startswith("_fastrag_user_custom_module_")
    }
    assert module_keys_after == module_keys_before
    assert sys.path == original_sys_path


@pytest.mark.asyncio
async def test_hook_loader_creates_unique_module_names(tmp_path):
    code_dir_one = tmp_path / "model_dir_one"
    code_dir_one.mkdir()
    (code_dir_one / "custom.py").write_text("""
async def load_model(code_dir):
    return "model_one"
""")

    code_dir_two = tmp_path / "model_dir_two"
    code_dir_two.mkdir()
    (code_dir_two / "custom.py").write_text("""
async def load_model(code_dir):
    return "model_two"
""")

    loader_one = HookLoader(str(code_dir_one))
    loader_two = HookLoader(str(code_dir_two))
    await loader_one.load()
    await loader_two.load()

    assert loader_one._module_name is not None
    assert loader_two._module_name is not None
    assert loader_one._module_name != loader_two._module_name


@pytest.mark.asyncio
async def test_hook_loader_creates_sync_adapter(tmp_path):
    code_dir = tmp_path / "model_dir_sync"
    code_dir.mkdir()
    custom_py = code_dir / "custom.py"
    custom_py.write_text(
        """
def load_model(code_dir):
    return "model"

def score(data, model, **kwargs):
    return [1]
"""
    )

    loader = HookLoader(str(code_dir), max_workers=2)
    adapter = await loader.load()

    assert isinstance(adapter, SyncModelAdapter)
    assert adapter._executor is not None
    assert adapter._executor._max_workers == 2
