import time

def log(msg: str, log_file=None):
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not log_file:
        print(f'({cur_time}) {msg}')
    else:
        with open(log_file, 'a') as f:
            f.write(f"({cur_time}) {msg}")
