import threading
import time


def load_model(code_dir):
    return {"thread_id": threading.get_ident()}


def score(data, model, **kwargs):
    time.sleep(0.01)  # encourage concurrency
    current_tid = threading.get_ident()
    return {"model_thread_id": model["thread_id"], "current_thread_id": current_tid}
