
from threading import Thread, Lock
import time
import queue
queue_var = queue.Queue()
# 定义一个准备作为线程任务的函数
# 创建互斥锁
mutex = Lock()


def action(max, threadname):
    for i in range(max):
        global mutex
        mutex.acquire()
        my_sum = queue_var.get()
        my_sum += 1
        queue_var.put(my_sum)
        print(threadname + ':  ' + str(my_sum) + "\r")
        mutex.release()
        time.sleep(1)
    print(threadname + " finish...... ")


if __name__ == '__main__':
    start = time.time()
    my_sum = 0
    queue_var.put(my_sum)
    threadlist = []
    for i in range(3):
        threadi = Thread(target=action, args=(20 + i * 10, "thread" + str(i + 1)))
        threadlist.append(threadi)
    for i in threadlist:
        i.start()
        time.sleep(1)
    for i in threadlist:
        i.join()

    end = time.time()
    print("Duration time: %0.3f s" % (end - start))
