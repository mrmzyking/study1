import threading
import serial
from threading import Thread, Lock
import time
import queue


class RS232Class(threading.Thread):
    def __init__(self, portname: str, baudrate: int, bytesize: int, parity: str, stopbits: float, timeout: float):
        threading.Thread.__init__(self)
        self.uartPort = portname
        self.uartBaudrate = baudrate
        self.uartBytesize = bytesize
        self.uartParity = parity
        self.uartStopbits = stopbits
        self.uarttimeout = timeout
        self.uartInterface = None
        self.RS232TxDQueue_Info = queue.Queue()  # 发送数据队列
        self.RS232RxDQueue_Info = queue.Queue()  # 接受数据队列
        self.RS232isConnectFlag = False
        self.uartTxRxLock = Lock()

    def getRS232TxDQueue_Info(self):
        return self.RS232TxDQueue_Info

    def getRS232RxDQueue_Info(self):
        return self.RS232RxDQueue_Info

    def run(self) -> None:
        self.InitUartPort()
        threadIsDisconnect = Thread(target=RS232Class.is_Port_Open, args=(self,))
        threadTxDDataToSer = Thread(target=RS232Class.RS232TxDData, args=(self,))
        threadRxDDataFromSer = Thread(target=RS232Class.RS232RxDData, args=(self,))
        threadDuleData = Thread(target=RS232Class.RS232DuleRecvData, args=(self,))
        threadIsDisconnect.start()
        threadTxDDataToSer.start()
        threadRxDDataFromSer.start()
        threadDuleData.start()
        threadIsDisconnect.join()
        threadTxDDataToSer.join()
        threadRxDDataFromSer.join()
        threadDuleData.join()

    def InitUartPort(self):
        try:
            self.uartInterface = serial.Serial(port=self.uartPort,
                                               baudrate=self.uartBaudrate,
                                               bytesize=self.uartBytesize,
                                               parity=self.uartParity,
                                               stopbits=self.uartStopbits,
                                               timeout=self.uarttimeout)
            self.RS232isConnectFlag = True
            print("Open ok : when open {} , has successed".format(self.uartPort))
        except Exception as e :
            self.RS232isConnectFlag = False
            print("Open Err : when open {} , has err {}".format(self.uartPort,e))
            time.sleep(5)

    def is_Port_Open(self):
        while True:
            if self.uartInterface is None :
                self.RS232TxDQueue_Info.empty()
                self.RS232RxDQueue_Info.empty()
                self.InitUartPort()
            else:
                if self.uartInterface.is_open :
                    self.RS232isConnectFlag = True
                else:
                    self.RS232isConnectFlag = False
                    self.RS232TxDQueue_Info.empty()
                    self.RS232RxDQueue_Info.empty()
                    self.uartInterface.close()
                    time.sleep(2)
                    self.InitUartPort()

    def RS232TxDData(self):
        while True:
            if self.uartInterface is None :
                time.sleep(2)
            else:
                if self.uartInterface.is_open and self.RS232TxDQueue_Info.not_empty:
                    try:
                        data = self.RS232TxDQueue_Info.get()
                        # self.uartTxRxLock.acquire()
                        self.uartInterface.write(data)
                        print("RS232Class : RS232 Client Info: TxD Data ({} bytes) from server !)".format(len(data)))
                        # self.uartTxRxLock.release()
                    except Exception as e:
                        self.RS232isConnectFlag = False
                        self.RS232TxDQueue_Info.empty()

    def RS232RxDData(self):
        while True:
            if self.uartInterface is None:
                time.sleep(2)
            else:
                if self.uartInterface.is_open :
                    try:
                        # self.uartTxRxLock.acquire()
                        data = self.uartInterface.readall()
                        # self.uartTxRxLock.release()
                        if len(data) != 0 :
                            print("RS232 Client Info: RxD Data ({} bytes) from server !)".format(len(data)))
                            self.RS232RxDQueue_Info.put(data)
                    except Exception as e:
                        self.RS232isConnectFlag = False
                        self.RS232TxDQueue_Info.empty()

    def RS232DuleRecvData(self):
        while True:
            if self.RS232RxDQueue_Info.not_empty :
                data = self.RS232RxDQueue_Info.get()
                self.RS232TxDQueue_Info.put(data)
                print("RS232Class : RS232DuleRecvData is in ......")


if __name__ == '__main__':
    u = RS232Class(portname="COM7", baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.2)
    u.start()
    u.join()
