import time
import warnings
from contextlib import contextmanager

import torch

class TimeCounter:
    """
    Usages:
        1. @TimeCounter.count_time 用于对函数进行计时
        2. with TimeCounter.profile_time 用于对中间某一程序段进行计时
        3. 直接调用 TimeCounter.profile_time 的__enter__ 和 __exit__用于对中间某一程序段进行计时

    Example:
    ```python
        @TimeCounter.count_time(warmup_interval=4)
        def fun1():
            time.sleep(2)

        @TimeCounter.count_time()
        def fun2():
            time.sleep(1)

        with TimeCounter.profile_time('sleep1'):
            print('start test profile_time')
            time.sleep(2)
            print('end test profile_time')

        # 第二种用法：直接在代码前后插入上下文，不需要对代码进行缩进
        time_counter = TimeCounter.profile_time('sleep3')
        time_counter.__enter__()
        pass
        time_counter.__exit__(None, None, None)
    """

    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls, log_interval=1, warmup_interval=1, with_sync=True):
        assert warmup_interval >= 1

        def _register(func):
            if func.__name__ in cls.names:
                raise RuntimeError(
                    'The registered function name cannot be repeated!')
            # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(
                count=0,
                pure_inf_time=0,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                pure_inf_time = cls.names[func.__name__]['pure_inf_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup_interval = cls.names[func.__name__]['warmup_interval']
                with_sync = cls.names[func.__name__]['with_sync']

                count += 1
                cls.names[func.__name__]['count'] = count

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                if count >= warmup_interval:
                    pure_inf_time += elapsed
                    cls.names[func.__name__]['pure_inf_time'] = pure_inf_time

                    if count % log_interval == 0:
                        times_per_count = 1000 * pure_inf_time / (
                            count - warmup_interval + 1)
                        print(
                            f'[{func.__name__}]-{count} times per count: '
                            f'{times_per_count:.1f} ms',
                            flush=True)

                return result

            return fun

        return _register

    @classmethod
    @contextmanager
    def profile_time(cls,
                     func_name,
                     log_interval=1,
                     warmup_interval=1,
                     with_sync=True):
        assert warmup_interval >= 1
        warnings.warn('func_name must be globally unique if you call '
                      'profile_time multiple times')

        if func_name in cls.names:
            count = cls.names[func_name]['count']
            pure_inf_time = cls.names[func_name]['pure_inf_time']
            log_interval = cls.names[func_name]['log_interval']
            warmup_interval = cls.names[func_name]['warmup_interval']
            with_sync = cls.names[func_name]['with_sync']
        else:
            count = 0
            pure_inf_time = 0
            cls.names[func_name] = dict(
                count=count,
                pure_inf_time=pure_inf_time,
                log_interval=log_interval,
                warmup_interval=warmup_interval,
                with_sync=with_sync)

        count += 1
        cls.names[func_name]['count'] = count

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        yield

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if count >= warmup_interval:
            pure_inf_time += elapsed
            cls.names[func_name]['pure_inf_time'] = pure_inf_time

            if count % log_interval == 0:
                times_per_count = 1000 * pure_inf_time / (
                    count - warmup_interval + 1)
                print(
                    f'[{func_name}]-{count} times per count: '
                    f'{times_per_count:.1f} ms',
                    flush=True)