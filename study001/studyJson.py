import json
import os
import time
import datetime

from appdirs import unicode


class JsonClass(object):
    def __init__(self, filepath: str):
        self.filePath = filepath
        self.jsonfilelist = []

    '''把时间戳转化为时间: 1479264792 to 2016-11-16 10:53:12'''
    def TimeStampToTime(self, timestamp):
        timeStruct = time.localtime(timestamp)
        return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)

    '''获取文件的访问时间'''
    def get_FileAccessTime(self, filePath):
        filePath = unicode(filePath, 'utf8')
        t = os.path.getatime(filePath)
        return self.TimeStampToTime(t)

    '''获取文件的创建时间'''
    def get_FileCreateTime(self, filePath):
        filePath = unicode(filePath, 'utf8')
        t = os.path.getctime(filePath)
        return self.TimeStampToTime(t)

    '''获取文件的修改时间'''
    def get_FileModifyTime(self, filePath):
        filePath = unicode(filePath, 'utf8')
        t = os.path.getmtime(filePath)
        return self.TimeStampToTime(t)

    def ScanPathFile(self):
        path_list = os.listdir(self.filePath)
        for filename in path_list:
            if os.path.splitext(filename)[1] == '.json':
                XXXXXX
