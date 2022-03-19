import time
import os
import psutil


def track(func):
    def wrapper(*args, **kwargs):
        cpu_before = psutil.cpu_percent(1)# / os.cpu_count()
        mem_before = psutil.Process(os.getpid()).memory_info().rss

        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = round(time.time() - start, 2)

        cpu_during = psutil.cpu_percent()# / os.cpu_count()
        mem_after = psutil.Process(os.getpid()).memory_info().rss
        print("{}: mem_consumed: {} runtime: {}, cpu_change: {}".format(func.__name__, mem_after - mem_before,
                                                                        elapsed_time, cpu_during - cpu_before))
        return {"result": result, "memory_before": mem_before, "memory_after": mem_after,
                "execution_time": elapsed_time, "cpu_before": cpu_before, "cpu_during": cpu_during}
    return wrapper

# @track
# def test():
#     for i in range(100):
#         sum([random.randrange(1, 100, 1) for i in range(1000)])
#     return