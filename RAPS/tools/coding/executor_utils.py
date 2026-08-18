"""Run a call under a wall-clock limit, so a candidate program that never returns cannot
stall the evaluation."""
from threading import Thread


class PropagatingThread(Thread):
    """A thread that re-raises in the caller whatever its target raised, and returns the
    target's value from join(), which a plain Thread discards. Both are set before the
    thread starts, so joining a target that has not finished reports the timeout rather
    than an attribute that was never assigned."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ret = None
        self.exc = None

    def run(self):
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def function_with_timeout(func, args, timeout):
    """`func(*args)` if it finishes within `timeout` seconds, else TimeoutError. The worker
    is a daemon, so a call that never returns is abandoned rather than left holding the
    interpreter open at exit."""
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError()
    return result_container[0]
