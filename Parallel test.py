import numpy as np
import time
from multiprocessing import Pool

def func(x):
    time.sleep(0.0005)
def func2(x):
    time.sleep(0.0005)
def func3(x):
    time.sleep(0.0005)


if __name__ == "__main__":
    data = np.random.randint(0, 100, 10000)
    print("start for part")
    start = time.time()
    for i in range(0, 10000):
        func(data[i])
    end = time.time()
    print("for time is {}".format(end - start))

    print("start line map part")
    start = time.time()
    result = map(func2, data)
    end = time.time()
    print("line map time is {}".format(end - start))

    print("start parallel part")
    start = time.time()
    pool = Pool(8)
    pool.map(func3, data)
    end = time.time()
    print("parallel time is {}".format(end - start))