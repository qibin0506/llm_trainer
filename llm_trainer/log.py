import time, os, atexit
from io import TextIOWrapper
from typing import Optional


def _get_log_dir() -> str:
    log_dir = os.environ.get('LOG_DIR', './log')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class Logger:
    def __init__(self, log_file_name = None, log_dir = None):
        self.log_file_name = log_file_name
        self.log_file: Optional[TextIOWrapper] = None

        if not log_dir:
            self.log_dir = _get_log_dir()
        else:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir

        self.flush_interval = int(os.environ.get('LOG_FLUSH_INTERVAL', '1'))
        self.log_steps = 0

    @staticmethod
    def std_log(msg: str):
        log_content = Logger._build_log(msg)
        print(log_content)

    def log(self, msg: str, log_to_console = True):
        log_content = Logger._build_log(msg)

        if log_to_console:
            print(log_content)

        if self._open_file():
            self.log_file.write(f'{log_content}\n')
            if self.log_steps % self.flush_interval == 0:
                self.log_file.flush()

        self.log_steps += 1
        return self

    def release(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    @staticmethod
    def _build_log(msg: str):
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return f'[{cur_time}] {msg}'

    def _open_file(self) -> bool:
        if not self.log_file_name:
            return False

        if self.log_file:
            return True

        self.log_file = open(os.path.join(self.log_dir, self.log_file_name), 'a', encoding='utf-8')
        atexit.register(self.release)

        return True
