import time, os

def get_log_dir() -> str:
    log_dir = os.environ['LOG_DIR']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    return f'{log_dir}/' if not log_dir.endswith('/') else log_dir


def log(msg: str, log_file=None):
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not log_file:
        print(f'[{cur_time}] {msg}')
    else:
        with open(log_file, 'a') as f:
            f.write(f"[{cur_time}] {msg}")
