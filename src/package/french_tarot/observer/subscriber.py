from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread


class Kill:
    pass


class Subscriber(ABC):
    def __init__(self):
        self._queue = Queue()
        self._thread = Thread(target=self.loop)

    def start(self):
        self.setup()
        self._thread.start()

    def stop(self):
        self.push(Kill())
        self._thread.join()
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def loop(self):
        run = True
        while run:
            message = self._queue.get()
            try:
                if not isinstance(message, Kill):
                    self.update(message)
                else:
                    run = False
            finally:
                self._queue.task_done()

    def push(self, data: any):
        self._queue.put(data)

    @abstractmethod
    def update(self, data: any):
        pass
