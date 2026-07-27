import pytest

from fastrag.loader import HookLoader


@pytest.mark.asyncio
async def test_hook_loader_mixed_hooks_error(tmp_path):
    code_dir = tmp_path / "model_dir_mixed"
    code_dir.mkdir()
    custom_py = code_dir / "custom.py"
    # Mixed hooks: async init, sync load_model
    custom_py.write_text("""
async def init(code_dir):
    pass

def load_model(code_dir):
    pass
""")
    loader = HookLoader(str(code_dir))
    with pytest.raises(
        ValueError, match="Hooks must be either all synchronous or all asynchronous"
    ):
        await loader.load()
