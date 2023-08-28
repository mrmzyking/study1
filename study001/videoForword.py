from datetime import datetime
from threading import Thread, Lock, Semaphore
import time
import queue

import cv2

class VideoForword(Thread):
    def __init__(self,fpath):
        Thread.__init__(self)
        self.videoPath = fpath
        # self.sp = Semaphore(25)
        self.readqueue = queue.Queue(maxsize=25)
        self.cap = None

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.videoPath)
        readThread = Thread(target=VideoForword.ReadVideoStream, args=(self,))
        openThread = Thread(target=VideoForword.PushVideoStream,args=(self,))
        moniThread = Thread(target=VideoForword.MonitorQuene, args=(self,))
        moniThread.start()
        readThread.start()
        openThread.start()
        moniThread.join()
        openThread.join()
        readThread.join()

    def MonitorQuene(self):
        while True:
            qsize = self.readqueue.qsize()
            print("{} Info : 当前队列中还有 {} 数据需要处理......".format(datetime.now(),qsize))
            time.sleep(0.03)

    def ReadVideoStream(self):
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while True :
            if self.cap.isOpened() is True:
                ret, frame = self.cap.read()
                if ret is not True:
                    # print("{} Err : no Frame has been catch......".format(datetime.now()))
                    time.sleep(0.04)
                else:
                    # print("{} Info : one Frame has been catch......".format(datetime.now()))
                    # self.sp.acquire()
                    # 使用背景减除器进行前景提取
                    fgmask = fgbg.apply(frame)
                    # 找到轮廓
                    contours,a1= cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # 在帧中绘制轮廓
                    cv2.drawContours(frame, contours, -1, ( 0, 255, 0), 1)
                    # 显示帧
                    # cv2.imshow('Frame', frame)
                    self.readqueue.put(frame)



    def PushVideoStream(self):
        while True :
            if self.cap.isOpened() is True and self.readqueue.not_empty:
                fram = self.readqueue.get()
                # self.sp.release()
                cv2.imshow("video", fram)
                time.sleep(0.04)
                cv2.waitKey(1)
            else:
                self.cap.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    u = VideoForword(fpath="C:\\Users\\mzy\\Desktop\\Screenrecorder-2023-08-07-16-15-21-78.mp4")
    u.start()
    u.join()