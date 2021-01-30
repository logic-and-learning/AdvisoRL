# Created by gaglione
import time
import functools
from contextlib import contextmanager

class Timer:

    def __init__(self):
        self.__total_elapsed = 0
        self.__session_start = None

    def __now(self):
        return time.perf_counter()


    def running(self):
        """Return `True` if the timer is activated."""
        return self.__session_start is not None

    def elapsed(self):
        """Return the total elapsed time wile the timer was activated."""
        if self.running():
            session_elapsed = self.__now() - self.__session_start
        else:
            session_elapsed = 0
        return self.__total_elapsed + session_elapsed

    def resume(self):
        """Activate the timer."""
        if not self.running():
            self.__session_start = self.__now()
        return self

    def start(self):
        """Same as `resume` but raises an error if already activated."""
        if self.running():
            raise RuntimeError("Timer already running.")
        return self.resume()

    def stop(self):
        """Desactivate the timer."""
        self.__total_elapsed = self.elapsed()
        self.__session_start = None
        return self

    def reset(self):
        """Reset the total elapsed time to zero. The timer stay activated if it was already running."""
        self.__total_elapsed = 0
        if self.running():
            self.__session_start = self.__now()
        return self


    def __enter__(self):
        """
            Timer can be used as a context manager.

            >>> with Timer() as t:
            >>>     ...
            >>> print(t.elapsed())
        """
        self.start()
        return self
    def __exit__(self, *args):
        """.. seealso:: __enter__()"""
        self.stop()

    def __call__(self, func):
        """
            Timer can be used as a decorator, and time will be incremented each time the function is called.

            >>> @timer
            >>> def foo(*args, **kwargs):
            >>>     ...
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    @contextmanager
    def include(self):
        """
            Ensure that the code calculation time is included in the measured time.

            >>> with timer.include():
            >>>     ...

            >>> @timer.include()
            >>> def foo(*args, **kwargs):
            >>>     ...
        """
        if not self.running():
            self.start()
            yield self
            self.stop()
        else:
            yield self

    @contextmanager
    def exclude(self):
        """
            Ensure that the code calculation time is excluded from the measured time.

            >>> with timer.exclude():
            >>>     ...

            >>> @timer.exclude()
            >>> def foo(*args, **kwargs):
            >>>     ...
        """
        if self.running():
            self.stop()
            yield self
            self.start()
        else:
            yield self

    def __float__(self):
        return float(self.elapsed())
    def __bool__(self):
        """A timer has a boolean value of `False` after complete reset."""
        return self.running() or self.elapsed()
    def __repr__(self):
        return "{!s}[{:.3f}s{}]".format(
            self.__class__.__name__,
            self.elapsed(),
            "..." if self.running() else "",
        )
    def __str__(self):
        # return "{:.3f}".format(
        #     self.elapsed()
        # )
        return repr(self)
