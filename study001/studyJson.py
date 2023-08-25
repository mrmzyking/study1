import json
import os
import time
import datetime
from typing import Any
from rs232totcp import *
from rs232toudp import *
from rs232study import *
from tcpclient import *
from udpclient import *
import socket
import threading
from threading import Thread, Lock
import time
import queue

from appdirs import unicode

"""
{
  "IPMode": "TCPClient",  # TCPClient or UDPClient
  "RS232": {
    "PortName": "COM1",   # COM1.....COMn
    "BaudRate": 9600,     # (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800,
                             9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000,
                             576000, 921600, 1000000, 1152000, 1500000, 2000000, 2500000,
                             3000000, 3500000, 4000000)
    "ByteSize": 8.0,      # 5,6,7,8
    "Parity": "N",        # 'N', PARITY_NONE
                            'E', PARITY_EVEN
                            'O', PARITY_ODD
                            'M', PARITY_MARK
                            'S',PARITY_SPACE
    "StopBits": 1.0,      # 1 , 1.5 , 2.0
    "TimeOut": 0.2         # 超时时间设置 0.01至1，单位s
  },
  "TCPClient": {
    "DesIP":"127.0.0.1",  # 服务端IP地址
    "DesPort": 4001       # 服务端端口号
  },
  "UDPClient":{
    "LocIP": "127.0.0.1", # 本机ip地址，主要针对都网口的情况，单网卡可以默认为127.0.0.1
    "DesIP": "127.0.0.1", # 服务端的ip地址
    "DesPort": 8888       # 服务端的端口号
  }
}
"""


class RS232InfoClass(object):
    def __init__(self):
        self.PortName: str = ""
        self.BaudRate: int = 9600
        self.ByteSize: int = 8
        self.Parity: str = 'N'
        self.StopBits: float = 1.0
        self.TimeOut: int = 20

    def setInfo(self, pname, br, bs, pr, sb, to):
        self.PortName: str = pname
        self.BaudRate: int = br
        self.ByteSize: int = bs
        self.Parity: str = pr
        self.StopBits: float = sb
        self.TimeOut: int = to

    def getInfo(self):
        return self.PortName, self.BaudRate, self.ByteSize, self.Parity, self.StopBits, self.TimeOut


class UDPClientInfoClass(object):
    def __init__(self):
        self.SorPort: int = 0
        self.DesIP: str = ""
        self.DesPort: int = 0

    def setInfo(self, sport, dip, dport):
        self.SorPort: str = sport
        self.DesIP: str = dip
        self.DesPort: int = dport

    def getInfo(self):
        return self.SorPort, self.DesIP, self.DesPort


class TCPClientInfoClass(object):
    def __init__(self):
        self.DesIP: str = ""
        self.DesPort: int = 0

    def setInfo(self, dip, dport):
        self.DesIP = dip
        self.DesPort = dport

    def getInfo(self):
        return self.DesIP, self.DesPort


class ConfigInfoClass(object):
    def __init__(self):
        self.IPMode: str = ""
        self.RS232: RS232InfoClass = RS232InfoClass()
        self.TCPClient: TCPClientInfoClass = TCPClientInfoClass()
        self.UDPClient: UDPClientInfoClass = UDPClientInfoClass()

    def setIPMode(self, ipmode: str):
        self.IPMode = ipmode

    def setRS232(self, pname, br, bs, pr, sb, to):
        self.RS232.setInfo(pname, br, bs, pr, sb, to)

    def setTCPClient(self, dip, dport):
        self.TCPClient.setInfo(dip, dport)

    def setUDPClient(self, lip, dip, dport):
        self.UDPClient.setInfo(lip, dip, dport)


class JsonFileFindClass(object):
    def __init__(self, filepath: str):
        self.filePath = filepath
        self.jsonfilelist = []

    '''把时间戳转化为时间: 1479264792 to 2016-11-16 10:53:12'''

    def TimeStampToTime(self, timestamp):
        timeStruct = time.localtime(timestamp)
        return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct), timestamp

    '''获取文件的访问时间'''

    def get_FileAccessTime(self, filePath):
        # filePath = unicode(filePath, 'utf8')
        t = os.path.getatime(filePath)
        return self.TimeStampToTime(t)

    '''获取文件的创建时间'''

    def get_FileCreateTime(self, filePath):
        # filePath = unicode(filePath, 'utf8')
        t = os.path.getctime(filePath)
        return self.TimeStampToTime(t)

    '''获取文件的修改时间'''

    def get_FileModifyTime(self, filePath):
        # filePath = unicode(filePath, 'utf8')
        t = os.path.getmtime(filePath)
        return self.TimeStampToTime(t)

    def ScanPathFile(self):
        path_list = os.listdir(self.filePath)
        jsonfilelist = []
        for filename in path_list:
            if os.path.splitext(filename)[1] == '.json':
                filepath = self.filePath + "\\" + filename
                modifytime, t = self.get_FileModifyTime(filePath=filepath)
                jsonfilelist.append([t, modifytime, filepath])
        print(jsonfilelist)
        return jsonfilelist

    def GetNewstJsonFile(self):
        filelist = self.ScanPathFile()
        time0 = None
        newfile = None
        if len(filelist) <= 0:
            newfile = None
        elif len(filelist) == 1:
            newfile = filelist[0][2]
        else:
            time0 = filelist[0][0]
            newfile = filelist[0][2]
            for i in filelist:
                if i[0] >= time0:
                    newfile = i[2]
        return newfile

    def AnalyzeIPMode(self, dictinfo: dict):
        rtval = None
        rtflag = False
        try:
            rtval = dictinfo["IPMode"]
            rtflag = True
        except Exception as e:
            print("Err : Json file has no IPMode , {}".format(e))
        return rtflag, rtval

    def AnalyzeRS232(self, dictinfo: dict):
        retflag = False
        retval = None
        dictval = None
        try:
            dictval = dictinfo["RS232"]
            retval = RS232InfoClass()
            retflag = True
        except Exception as e:
            retflag = False
            retval = None
            print("Err : Json file has no RS232".format(e))
        if retflag is True:
            try:
                pname = dictval["PortName"]
                br = dictval["BaudRate"]
                bs = dictval["ByteSize"]
                pr = dictval["Parity"]
                sb = dictval["StopBits"]
                to = dictval["TimeOut"]
                retval.setInfo(pname, br, bs, pr, sb, to)
                retflag = True
            except Exception as e:
                retflag = False
                retval = None
                print(
                    "Err : Json file(RS232 {}) has no item(PortName,BaudRate,ByteSize,Parity,StopBits,TimeOut ) {}".format(
                        dictval, e))
        return retflag, retval

    def AnalyzeTCPClient(self, dictinfo: dict):
        retflag = False
        retval = None
        dictval = None
        try:
            dictval = dictinfo["TCPClient"]
            retval = TCPClientInfoClass()
            retflag = True
        except Exception as e:
            retflag = False
            retval = None
            print("Err : Json file has no TCPClient".format(e))
        if retflag is True:
            try:
                dip = dictval["DesIP"]
                dport = dictval["DesPort"]
                retval.setInfo(dip, dport)
                retflag = True
            except Exception as e:
                retflag = False
                retval = None
                print(
                    "Err : Json file(TCPClient {}) has no item(DesIP,DesPort) {}".format(
                        dictval, e))
        return retflag, retval

    def AnalyzeUDPClient(self, dictinfo: dict):
        retflag = False
        retval = None
        dictval = None
        try:
            dictval = dictinfo["UDPClient"]
            retval = UDPClientInfoClass()
            retflag = True
        except Exception as e:
            retflag = False
            retval = None
            print("Err : Json file has no UDPClient".format(e))
        if retflag is True:
            try:
                sport = dictval["SorPort"]
                dip = dictval["DesIP"]
                dport = dictval["DesPort"]
                retval.setInfo(sport, dip, dport)
                retflag = True
            except Exception as e:
                retflag = False
                retval = None
                print(
                    "Err : Json file(UDPClient {}) has no item(DesIP,DesPort) {}".format(
                        dictval, e))
        return retflag, retval

    def AnanyzlyJsonFile(self):
        newfile = self.GetNewstJsonFile()
        load_dict = None
        retval = None
        if newfile is None:
            print("Err  : No Jsonfile has been found!")
        else:
            # 根据返回数据进行文件打开和读取操作
            try:
                with open(newfile, 'r') as load_f:
                    load_dict = json.load(load_f)
            except Exception as e:
                print("Err : File open fault: {}".format(e))
            print(load_dict)
            flag1, retval1 = self.AnalyzeIPMode(load_dict)
            flag2, retval2 = self.AnalyzeRS232(load_dict)
            flag3, retval3 = self.AnalyzeTCPClient(load_dict)
            flag4, retval4 = self.AnalyzeUDPClient(load_dict)
            if flag1 is True and flag2 is True and flag3 is True and flag4 is True:
                retval = ConfigInfoClass()
                retval.IPMode = retval1
                retval.RS232 = retval2
                retval.TCPClient = retval3
                retval.UDPClient = retval4
            return retval


if __name__ == '__main__':
    a = JsonFileFindClass("D:\python_prj\study\study001")
    val = a.AnanyzlyJsonFile()
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
        print("Err : only support TCPClient and UDPClient")
