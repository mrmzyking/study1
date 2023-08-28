import socket
from datetime import datetime
from threading import Thread, Lock
import time
import queue
import IPy


class udpclientsocket(Thread):
    def __init__(self, dip: str, dport: int, sport: int):
        """
        初始化
        :param dip: udp目标IP地址
        :param dport: udp目标端口号
        :param sport: 本地UDP绑定端口号
        """
        Thread.__init__(self)
        self.serveripaddr = dip
        self.serveripport = dport
        self.clientipport = sport
        self.serversocket = None
        self.isConnectFlag = False
        self.UDPTxDQueue_Info = queue.Queue()  # 发送数据队列
        self.UDPRxDQueue_Info = queue.Queue()  # 接受数据队列
        self.RxdTime = None
        self.TxdTime = None

    def getUDPTxDQueue_Info(self):
        return self.UDPTxDQueue_Info
    def getUDPRxDQueue_Info(self):
        return self.UDPRxDQueue_Info

    def run(self) -> None:
        self.UDPconnectServer()
        threadIsDisconnect = Thread(target=udpclientsocket.UDPIsServerDisconnect, args=(self,))
        threadTxDDataToSer = Thread(target=udpclientsocket.UDPSendDataToServer, args=(self,))
        threadRxDDataFromSer = Thread(target=udpclientsocket.UDPRecvServerData, args=(self,))
        threadDuleData = Thread(target=udpclientsocket.UDPDuleRecvData, args=(self,))
        threadIsDisconnect.start()
        threadTxDDataToSer.start()
        threadRxDDataFromSer.start()
        threadDuleData.start()
        threadIsDisconnect.join()
        threadTxDDataToSer.join()
        threadRxDDataFromSer.join()
        threadDuleData.join()

    def is_ip(self, address):
        """
        判定当前IP地址是否有错误
        :param address:
        :return: 无错误返回真，有错误返回假
        """
        try:
            IPy.IP(address)
            return True
        except Exception as e:
            return False

    def is_port(self, port):
        """
        判定当前端口号是否有错误
        :param port: 端口号
        :return: 若无错误则为真，有错误则为假
        """
        OK = False
        try:
            if int(port) in range(0, 65536):
                OK = True
            else:
                OK = False
        except Exception as e:
            OK = False
        return OK

    def UDPconnectServer(self):
        """
        连接UDP服务端
        :return:
        """
        # 1.检查输入得IP地址是否正确
        flag1 = self.is_ip(self.serveripaddr)
        # 2.检查输入得端口号是否正确
        flag2 = self.is_port(self.serveripport)
        if (flag1 is True) and (flag2 is True):
            if self.serversocket is not None:
                self.serversocket.close()
            try:
                self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.serversocket.bind(("127.0.0.1", self.clientipport))
                # socket.AF_INET: 网络类型，IPv4
                # socket.SOCK_DGRAM： UDP 客户端
                self.isConnectFlag = True
                print("{} Client Info: Connect Server({},{}) successed......".format(datetime.now(),self.serveripaddr, self.serveripport))
            except Exception as e:
                print("{} Client Info: Connect Server({},{}) failed.:{}".format(datetime.now(),self.serveripaddr, self.serveripport, e))
                self.isConnectFlag = False
                self.serversocket.close()
        else:
            if flag1 is True:
                print("{} Client Info: Connect Server({},{}) failed......(server ip addr is wrong!)".format(
                    datetime.now(),self.serveripaddr, self.serveripport))
            if flag2 is True:
                print("{} Client Info: Connect Server({},{}) failed......(server ip port is wrong!)".format(
                    datetime.now(),self.serveripaddr, self.serveripport))

    def UDPIsServerDisconnect(self):
        """
        UDP断线重连
        :return:
        """
        while True:
            if self.isConnectFlag is False:
                self.UDPconnectServer()
                time.sleep(5)
            else:
                time.sleep(5)

    def UDPRecvServerData(self):
        while True:
            if self.isConnectFlag is True:
                try:
                    response, address = self.serversocket.recvfrom(1024)
                    self.UDPRxDQueue_Info.put(response)
                    print("{} UDP Client Info: Recv Data ({} bytes) from server {}!)".format(datetime.now(),len(response), address))
                except Exception as e:
                    self.isConnectFlag = False

    def UDPSendDataToServer(self):
        while True:
            if self.UDPTxDQueue_Info.not_empty:
                txddata = self.UDPTxDQueue_Info.get()
                try:
                    self.serversocket.sendto(txddata, (self.serveripaddr, self.serveripport))
                    print("{} UDP Client Info: Send Data ({} bytes) to server !)".format(datetime.now(),len(txddata)))
                except Exception as e:
                    self.isConnectFlag = False

    def UDPDuleRecvData(self):
        while True:
            if self.UDPRxDQueue_Info.not_empty:
                data = self.UDPRxDQueue_Info.get()
                self.UDPTxDQueue_Info.put(data)
                print("{} udpclientsocket  : UDPDuleRecvData is in ......".format(datetime.now(),))


if __name__ == '__main__':
    tcp = udpclientsocket(dip="127.0.0.1", dport=4001, sport=7777)
    tcp.start()
    tcp.join()
