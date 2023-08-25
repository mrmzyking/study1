import socket
import threading
from threading import Thread, Lock
import time
import queue
import IPy


class tcpclientsocket(threading.Thread):
    def __init__(self, sip: str, sport: int):
        threading.Thread.__init__(self)
        self.serveripaddr = sip
        self.serveripport = sport
        self.serversocket = None
        self.TCPisConnectFlag = False
        self.TCPTxDQueue_Info = queue.Queue() # 发送数据队列
        self.TCPRxDQueue_Info = queue.Queue() # 接受数据队列
        self.RxdTime = None
        self.TxdTime = None

    def TCPgetTCPTxDQueue_Info(self):
        return self.TCPTxDQueue_Info

    def TCPgetTCPRxDQueue_Info(self):
        return self.TCPRxDQueue_Info


    def run(self) -> None:
        tcpthreadIsDisconnect = Thread(target=tcpclientsocket.TCPIsServerDisconnect, args=(self,))
        tcpthreadTxDDataToSer = Thread(target=tcpclientsocket.TCPSendDataToServer, args=(self,))
        tcpthreadRxDDataFromSer = Thread(target=tcpclientsocket.TCPRecvServerData, args=(self,))
        tcpthreadDuleData = Thread(target=tcpclientsocket.TCPDuleRecvData, args=(self,))
        tcpthreadIsDisconnect.start()
        tcpthreadTxDDataToSer.start()
        tcpthreadRxDDataFromSer.start()
        tcpthreadDuleData.start()
        tcpthreadIsDisconnect.join()
        tcpthreadTxDDataToSer.join()
        tcpthreadRxDDataFromSer.join()
        tcpthreadDuleData.join()

    def is_ip(self, address):
        try:
            IPy.IP(address)
            return True
        except Exception as e:
            return False

    def is_port(self, port):
        OK = False
        try:
            if int(port) in range(0, 65536):
                OK = True
            else:
                OK = False
        except Exception as e:
            OK = False
        return OK

    def TCPconnectServer(self):
        # 1.检查输入得IP地址是否正确
        flag1 = self.is_ip(self.serveripaddr)
        # 2.检查输入得端口号是否正确
        flag2 = self.is_port(self.serveripport)
        if (flag1 is True) and (flag2 is True):
            if self.serversocket is not None :
                self.serversocket.close()
            self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.serversocket.connect((self.serveripaddr, self.serveripport))
                self.TCPisConnectFlag = True
                print("Client Info: Connect Server({},{}) successed......".format(self.serveripaddr, self.serveripport))
            except Exception as e :
                print("Client Info: Connect Server({},{}) failed.:{}".format(self.serveripaddr, self.serveripport,e))
                self.TCPisConnectFlag = False
                self.serversocket.close()
        else:
            if flag1 is True:
                print("Client Info: Connect Server({},{}) failed......(server ip addr is wrong!)".format(
                    self.serveripaddr, self.serveripport))
            if flag2 is True:
                print("Client Info: Connect Server({},{}) failed......(server ip port is wrong!)".format(
                    self.serveripaddr, self.serveripport))

    def TCPIsServerDisconnect(self):
        while True:
            if self.TCPisConnectFlag is False:
                self.TCPconnectServer()
                time.sleep(1)
            else:
                time.sleep(1)

    def TCPRecvServerData(self):
        while True:
            if self.TCPisConnectFlag is False:
                time.sleep(1)
            else:
                try:
                    recvdata = self.serversocket.recv(1024)
                    if len(recvdata) > 0:
                        self.RxdTime = time.time()
                        datalens = len(recvdata)
                        print("Client Info: Recv Data ({} bytes) from server!)".format(datalens))
                        self.TCPRxDQueue_Info.put(recvdata)
                except Exception as e:
                    self.TCPRxDQueue_Info.empty()
                    print("Client Info: Recv Data ({},{}) failed.:{}".format(self.serveripaddr, self.serveripport, e))
                    self.TCPisConnectFlag = False

    def TCPSendDataToServer(self):
        while True :
            if self.TCPisConnectFlag is False:
                time.sleep(1)
            else:
                try:
                    if self.TCPTxDQueue_Info.not_empty :
                        data = self.TCPTxDQueue_Info.get()
                        datalens = len(data)
                        self.serversocket.send(data)
                        self.TxdTime = time.time()
                        print("Client Info: Transport Data ({} bytes) from server !)".format(datalens))
                except Exception as e:
                    print("Client Info: Txd Data ({},{}) failed.:{}".format(self.serveripaddr, self.serveripport, e))
                    self.TCPTxDQueue_Info.empty()
                    self.TCPisConnectFlag = False


    def TCPDuleRecvData(self):
        while True:
            if self.TCPRxDQueue_Info.not_empty :
                data = self.TCPRxDQueue_Info.get()
                self.TCPTxDQueue_Info.put(data)


if __name__ == '__main__':
    tcp = tcpclientsocket(sip="127.0.0.1", sport=12110)
    tcp.start()
    tcp.join()

