import time

from study001.rs232study import *
from study001.udpclient import *


class RS232ToUDP(udpclientsocket, RS232Class):
    def __init__(self, dip: str, dport: int, sport: int, portname: str, baudrate: int, bytesize: int, parity: str,
                 stopbits: float, timeout: float):
        udpclientsocket.__init__(self, dip=dip,
                                 dport=dport,
                                 sport=sport)
        RS232Class.__init__(self, portname=portname,
                            baudrate=baudrate,
                            bytesize=bytesize,
                            parity=parity,
                            stopbits=stopbits,
                            timeout=timeout)

    def run(self) -> None:
        self.UDPconnectServer()
        self.InitUartPort()
        udpthreadIsDisconnect = Thread(target=udpclientsocket.UDPIsServerDisconnect, args=(self,))
        udpthreadTxDDataToSer = Thread(target=udpclientsocket.UDPSendDataToServer, args=(self,))
        udpthreadRxDDataFromSer = Thread(target=udpclientsocket.UDPRecvServerData, args=(self,))
        udpthreadDuleData = Thread(target=RS232ToUDP.UDPDuleRecvData, args=(self,))
        rs232threadIsDisconnect = Thread(target=RS232Class.is_Port_Open, args=(self,))
        rs232threadTxDDataToSer = Thread(target=RS232Class.RS232TxDData, args=(self,))
        rs232threadRxDDataFromSer = Thread(target=RS232Class.RS232RxDData, args=(self,))
        rs232threadDuleData = Thread(target=RS232ToUDP.RS232DuleRecvData, args=(self,))
        rs232threadIsDisconnect.start()
        rs232threadTxDDataToSer.start()
        rs232threadRxDDataFromSer.start()
        rs232threadDuleData.start()
        udpthreadIsDisconnect.start()
        udpthreadTxDDataToSer.start()
        udpthreadRxDDataFromSer.start()
        udpthreadDuleData.start()
        udpthreadIsDisconnect.join()
        udpthreadTxDDataToSer.join()
        udpthreadRxDDataFromSer.join()
        udpthreadDuleData.join()
        rs232threadIsDisconnect.join()
        rs232threadTxDDataToSer.join()
        rs232threadRxDDataFromSer.join()
        rs232threadDuleData.join()

    def RS232DuleRecvData(self, ):
        while True:
            # if RS232Class.getRS232RxDQueue_Info(self).not_empty :
            #     data = RS232Class.getRS232RxDQueue_Info(self).get()
            #     udpclientsocket.getUDPTxDQueue_Info(self).put(data)
            #     print("RS232ToUDP : RS232DuleRecvData is in ......")
            if RS232Class.getRS232RxDQueue_Info(self).not_empty :
                data = RS232Class.getRS232RxDQueue_Info(self).get()
                udpclientsocket.getUDPTxDQueue_Info(self).put(data)
                print("{} RS232ToUDP : RS232DuleRecvData is in ......".format(datetime.now(),))

    def UDPDuleRecvData(self, ):
        while True :
            if udpclientsocket.getUDPRxDQueue_Info(self).not_empty :
                data = udpclientsocket.getUDPRxDQueue_Info(self).get()
                RS232Class.getRS232TxDQueue_Info(self).put(data)
                print("{} RS232ToUDP : UDPDuleRecvData is in ......".format(datetime.now(),))


if __name__ == '__main__':
    c = RS232ToUDP(dip="127.0.0.1",
                   dport=4001,
                   sport=7777,
                   portname="COM7",
                   baudrate=9600,
                   bytesize=8,
                   parity='N',
                   stopbits=1,
                   timeout=0.2
                   )
    c.start()
    c.join()