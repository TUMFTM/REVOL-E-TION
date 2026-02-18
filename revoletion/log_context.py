from contextvars import ContextVar
from contextlib import contextmanager


log_context: ContextVar[str] = ContextVar("log_context", default="")


@contextmanager
def log_context_manager(name: str):
    token = log_context.set(name)
    try:
        yield
    finally:
        log_context.reset(token)
