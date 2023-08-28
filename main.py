# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
# import json
# import os
# import time
# import datetime
# from typing import Any
# from study001.rs232totcp import *
# from study001.rs232toudp import *
# from study001.rs232study import *
# from study001.tcpclient import *
# from study001.udpclient import *
from study001.studyJson import *
# import socket
# import threading
# from threading import Thread, Lock
# import time
# import queue
if __name__ == '__main__':
    foldpath = os.path.abspath('.')
    print("{} Info : Current Path = {}".format(datetime.now(),foldpath))
    a = JsonFileFindClass(foldpath)
    val = a.AnanyzlyJsonFile()
    if val is None:
        print("{} Err : when start app ,find no json file!".format(datetime.now()))
    else:
        if val.IPMode == "TCPClient":
            c = RS232ToTCP(dip=val.TCPClient.DesIP,
                                      dport=val.TCPClient.DesPort,
                                      portname=val.RS232.PortName,
                                      baudrate=val.RS232.BaudRate,
                                      bytesize=val.RS232.ByteSize,
                                      parity=val.RS232.Parity,
                                      stopbits=val.RS232.StopBits,
                                      timeout=val.RS232.TimeOut)
            c.start()
            c.join()
        elif val.IPMode == "UDPClient":
            c = RS232ToUDP(dip=val.UDPClient.DesIP,
                           dport=val.UDPClient.DesPort,
                           sport=val.UDPClient.SorPort,
                           portname=val.RS232.PortName,
                           baudrate=val.RS232.BaudRate,
                           bytesize=val.RS232.ByteSize,
                           parity=val.RS232.Parity,
                           stopbits=val.RS232.StopBits,
                           timeout=val.RS232.TimeOut
                           )
            c.start()
            c.join()
        else:
            print("{} Err : only support TCPClient and UDPClient".format(datetime.now(),))


