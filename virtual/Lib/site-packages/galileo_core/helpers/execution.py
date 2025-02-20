from asyncio import new_event_loop, run_coroutine_threadsafe
from threading import Thread
from typing import Any, Awaitable, Coroutine, TypeVar

T = TypeVar("T")


class AsyncExecutor:
    # Copied from: https://gist.github.com/gsakkis/18bc444607a590fe3f084a77aa54b4c2
    def __enter__(self) -> "AsyncExecutor":
        self._loop = new_event_loop()
        self._looper = Thread(target=self._loop.run_forever, daemon=True)
        self._looper.start()
        return self

    def __call__(self, coro: Awaitable[T]) -> T:
        return run_coroutine_threadsafe(coro, self._loop).result()

    def __exit__(self, *exc_info: Any) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._looper.join()
        self._loop.close()


def async_run(coroutine: Coroutine) -> Any:
    """
    Run an async coroutine synchronously.

    This function is useful for running async code in a synchronous context. This will
    ensure that the async code is run in a separate thread and the result is returned
    to the caller.

    Parameters
    ----------
    coroutine : Coroutine
        The coroutine to run.

    Returns
    -------
    Any
        The result of the coroutine.
    """
    with AsyncExecutor() as async_executor:
        return async_executor(coroutine)
