import os
import datetime
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, track
from functools import wraps
from typing import Optional
import logging
from rich.logging import RichHandler
from logging import FileHandler
# mac 系统上 pytorch 和 matplotlib 在 jupyter 中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def printbar():
    """打印时间
    """
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)


def pbar(container, totol: Optional[float] = None, description='Working...', transient: bool = False):
    """Example Usage of Rich Progress

    Basic Usage:

    ```python
    @pbar(list(range(100)))
    def do(i):
        print(i)
        time.sleep(0.2)
    ```

    Tutorial 1:

    ```python
    n = 100
    for i in track(range(n), description="Processing..."):
        time.sleep(0.2)
    ```

    Tutorial 2:

    ```python
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:", TimeElapsedColumn(), transient=True) as progress:
        task1 = progress.add_task("[red]Downloading...", total=1000)
        task2 = progress.add_task("[green]Processing...", total=1000)
        task3 = progress.add_task("[cyan]Cooking...", total=1000)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.3)
            progress.update(task3, advance=0.9)
            time.sleep(0.02)
    ```

    References:
        - [Rich Doc](https://rich.readthedocs.io/en/stable/progress.html#advanced-usage)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:", TimeElapsedColumn(),
                          transient=transient) as progress:
                for i in progress.track(container, description=description, total=totol):
                    func(i, *args, **kwargs)

        return wrapper

    return decorator


def get_logger(name: Optional[str] = None, filename: Optional[str] = None, level: str = 'NOTSET'):
    """获取一个 Rich 美化的 Logger"""
    name = name if name else __name__

    handlers = [RichHandler(
            rich_tracebacks=True,
        )]
    if filename:
        handlers.append(FileHandler(filename))

    logging.basicConfig(
        level=level,
        format='%(name)s: %(message)s',
        handlers=handlers)
    return logging.getLogger(name)


def run_once(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


