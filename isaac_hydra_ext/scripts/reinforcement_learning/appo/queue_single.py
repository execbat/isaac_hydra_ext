import multiprocessing as mp
import queue as pyqueue
from contextlib import suppress

class LatestQueue:
    def __init__(self):
        self._q = mp.Queue(maxsize=1)
        self._lock = mp.Lock()  

    def put_latest(self, item):
        with self._lock:
            while True:
                with suppress(pyqueue.Empty):
                    self._q.get_nowait()
                    continue
                break

            with suppress(pyqueue.Full):
                self._q.put_nowait(item)

    def get_latest(self, timeout=None): # blocking

        first = self._q.get(timeout=timeout)
        latest = first

        while True:
            with suppress(pyqueue.Empty):
                latest = self._q.get_nowait()
                continue
            break
        return latest

    def try_get_latest(self): # non blocking

        try:
            item = self._q.get_nowait()
        except pyqueue.Empty:
            return None
        latest = item
        while True:
            with suppress(pyqueue.Empty):
                latest = self._q.get_nowait()
                continue
            break
        return latest

