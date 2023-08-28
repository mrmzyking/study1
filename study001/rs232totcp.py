import time

from study001.rs232study import *
from study001.tcpclient import *


class RS232ToTCP(tcpclientsocket, RS232Class):
    def __init__(self, dip: str, dport: int,  portname: str, baudrate: int, bytesize: int, parity: str,
                 stopbits: float, timeout: float):
        tcpclientsocket.__init__(self, sip=dip,
                                 sport=dport)
        RS232Class.__init__(self, portname=portname,
                            baudrate=baudrate,
                            bytesize=bytesize,
                            parity=parity,
                            stopbits=stopbits,
                            timeout=timeout)

    def run(self) -> None:
        self.InitUartPort()
        tcpthreadIsDisconnect = Thread(target=tcpclientsocket.TCPIsServerDisconnect, args=(self,))
        tcpthreadTxDDataToSer = Thread(target=tcpclientsocket.TCPSendDataToServer, args=(self,))
        tcpthreadRxDDataFromSer = Thread(target=tcpclientsocket.TCPRecvServerData, args=(self,))
        tcpthreadDuleData = Thread(target=RS232ToTCP.TCPDuleRecvData, args=(self,))
        rs232threadIsDisconnect = Thread(target=RS232Class.is_Port_Open, args=(self,))
        rs232threadTxDDataToSer = Thread(target=RS232Class.RS232TxDData, args=(self,))
        rs232threadRxDDataFromSer = Thread(target=RS232Class.RS232RxDData, args=(self,))
        rs232threadDuleData = Thread(target=RS232ToTCP.RS232DuleRecvData, args=(self,))
        rs232threadIsDisconnect.start()
        rs232threadTxDDataToSer.start()
        rs232threadRxDDataFromSer.start()
        rs232threadDuleData.start()
        tcpthreadIsDisconnect.start()
        tcpthreadTxDDataToSer.start()
        tcpthreadRxDDataFromSer.start()
        tcpthreadDuleData.start()
        tcpthreadIsDisconnect.join()
        tcpthreadTxDDataToSer.join()
        tcpthreadRxDDataFromSer.join()
        tcpthreadDuleData.join()
        rs232threadIsDisconnect.join()
        rs232threadTxDDataToSer.join()
        rs232threadRxDDataFromSer.join()
        rs232threadDuleData.join()

    def RS232DuleRecvData(self, ):
        while True:
            if RS232Class.getRS232RxDQueue_Info(self).not_empty :
                data = RS232Class.getRS232RxDQueue_Info(self).get()
                tcpclientsocket.TCPgetTCPTxDQueue_Info(self).put(data)
                print("{} RS232ToUDP : RS232DuleRecvData is in ......".format(datetime.now(),))

    def TCPDuleRecvData(self, ):
        while True :
            if tcpclientsocket.TCPgetTCPRxDQueue_Info(self).not_empty :
                data = tcpclientsocket.TCPgetTCPRxDQueue_Info(self).get()
                RS232Class.getRS232TxDQueue_Info(self).put(data)
                print("{} RS232ToUDP : UDPDuleRecvData is in ......".format(datetime.now(),))


if __name__ == '__main__':
    c = RS232ToTCP(dip="127.0.0.1",
                   dport=4001,
                   portname="COM7",
                   baudrate=9600,
                   bytesize=8,
                   parity='N',
                   stopbits=1,
                   timeout=0.2
                   )
    c.start()
    c.join()