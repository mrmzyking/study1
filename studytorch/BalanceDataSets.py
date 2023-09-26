import math

import cv2
import os
import hashlib
import time
from random import sample
from studytorch.tuxiangengqiang import *


class DataSetFenLei(object):
    def __init__(self):
        self.TypeName = ""
        self.TypeNums = 0
        self.TypePath = []


class BalanceDataSet(object):
    def __init__(self, DatasetTxtPath, MaxDatasetNum):
        self.DatasetTxtPath = DatasetTxtPath
        self.MaxDatasetNum = MaxDatasetNum
        self.TypeNameList = []
        self.DataSetFenLeiList = []
        self.Before = {}
        self.After = {}

    def TongJiBefore(self):
        for i in self.DataSetFenLeiList:
            if isinstance(i,DataSetFenLei):
                self.Before[i.TypeName] = i.TypeNums

    def TongJiAfter(self):
        for i in self.DataSetFenLeiList:
            if isinstance(i,DataSetFenLei):
                self.After[i.TypeName] = i.TypeNums
    def ReadDataset(self):
        f = open(self.DatasetTxtPath)
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.replace("  ", " ")
            strlist = line.split(" ")
            typename = strlist[1]
            typepath = strlist[0]
            if typename in self.TypeNameList:
                # 1. 查找当前typename的位置
                datasetfenlei = self.FindDataSetFenLei(typename)
                # 2. 获取当前DataSetFenLeiList
                datasetfenlei.TypePath.append(typepath)
                # 3. 修改DataSetFenLeiList
                datasetfenlei.TypeNums = len(datasetfenlei.TypePath)
            else:
                self.TypeNameList.append(typename)
                tmp = DataSetFenLei()
                tmp.TypeName = typename
                tmp.TypePath.append(typepath)
                tmp.MaxDatasetNum = len(tmp.TypePath)
                self.DataSetFenLeiList.append(tmp)
        f.close()
        self.TongJiBefore()

    def FindDataSetFenLei(self, typename):
        retval = None
        for i in self.DataSetFenLeiList:
            if isinstance(i, DataSetFenLei):
                if i.TypeName == typename:
                    retval = i
                    break
        return retval

    def BananceDataset(self):
        datanums = []
        for i in self.DataSetFenLeiList:
            datanums.append(i.TypeNums)
        maxnum = max(datanums)
        maxnum = max(maxnum,self.MaxDatasetNum)
        for i in self.DataSetFenLeiList:
            if isinstance(i, DataSetFenLei):
                neednum = maxnum - i.TypeNums
                beishu = float(neednum / i.TypeNums)
                if beishu <= 1.0:
                    # 选择其中照片进行固定旋转3次
                    xuanzeshuju = int(neednum / 3)
                    xuanzetupian = sample(i.TypePath, xuanzeshuju)
                    for j in xuanzetupian:
                        # 1. 读取图片
                        img = self.ReadImage(j)
                        # 2. 旋转三次
                        QuanJiaoDu = []
                        for ll in range(1, 359, 1):
                            QuanJiaoDu.append(ll)
                        for k in range(3):
                            jiaodu = sample(QuanJiaoDu, 1)
                            tmpimg = self.XuanZhuanImg(img, jiaodu[0])
                            QuanJiaoDu.remove(jiaodu[0])
                            # 3. 获取图片保存路径
                            newpath = self.getImageFilePath(j)
                            # 4. 生成图片保存名称
                            filename = self.getImageFileName(j, k)
                            # 5. 保存图片
                            totlepath = newpath + filename
                            cv2.imwrite(totlepath,tmpimg)
                            # 6. 将保存图片信息保存到列表中
                            i.TypePath.append(totlepath)
                            i.TypeNums = len(i.TypePath)
                else:
                    # 每一张照片旋转几次进行存储
                    yuanshiliebiao = i.TypePath.copy()
                    for j in yuanshiliebiao:
                        # 1. 读取图片
                        # print(j)
                        img = self.ReadImage(j)
                        xuanzhuancishu = math.ceil(beishu)
                        # 2. 旋转xuanzhuancishu次
                        QuanJiaoDu = []
                        for ll in range(1, 359, 1):
                            QuanJiaoDu.append(ll)
                        for k in range(xuanzhuancishu):
                            jiaodu = sample(QuanJiaoDu, 1)
                            tmpimg = self.XuanZhuanImg(img, jiaodu[0])
                            QuanJiaoDu.remove(jiaodu[0])
                            # 3. 获取图片保存路径
                            newpath = self.getImageFilePath(j)
                            # 4. 生成图片保存名称
                            filename = self.getImageFileName(j, k)
                            # 5. 保存图片
                            totlepath = newpath + filename
                            cv2.imwrite(totlepath, tmpimg)
                            # 6. 将保存图片信息保存到列表中
                            i.TypePath.append(totlepath)
                            i.TypeNums = len(i.TypePath)
        self.TongJiAfter()

    def ReadImage(self, path):
        img = cv2.imread(path)
        return img

    def XuanZhuanImg(self, img, jiaodu):
        img2 = rotate_bound(img, jiaodu)
        return img2

    def getImageFilePath(self, path):
        retpath = ""
        if isinstance(path, str):
            tmppath = path
            tmplist = tmppath.split("\\")
            newpath = tmplist[0:-1]
            for i in newpath:
                retpath += i + "\\"
        return retpath

    def getImageFileName(self, path, num):
        retname = ""
        if isinstance(path, str):
            tmppath = path
            tmplist = tmppath.split("\\")
            oldname = tmplist[-1]
            name1 = oldname.split(".")
            name1[0] = name1[0] + str(num)
            retname = name1[0] + "." + name1[1]
        return retname

    def RewriteDataSetFile(self):
        try:
            with open(self.DatasetTxtPath, mode="wb+") as file:
                for i in self.DataSetFenLeiList:
                    if isinstance(i,DataSetFenLei):
                        typename = i.TypeName
                        for j in i.TypePath :
                            typepath = j
                            writestr = typepath + " " + str(typename) + "\n"
                            writestr = writestr.encode()
                            file.write(writestr)
        except Exception as e :
            print("err : 当写入数据集时，出现写入错误！{}".format(e))
        finally:
            file.close()

    def PrintString(self):
        print("数据增强之前的数据集为")
        print(self.Before)
        print("数据增强之后的数据集为")
        print(self.After)


if __name__ == '__main__':

    traindataset = BalanceDataSet("D:\\python_prj\\study\\studytorch\\shudan\\shudian\\datasets\\datasets.txt", 3000)

    traindataset.ReadDataset()
    print(traindataset.TypeNameList)
    traindataset.BananceDataset()
    traindataset.RewriteDataSetFile()
    traindataset.PrintString()

    # testdataset = BalanceDataSet("D:\\python_prj\\study\\studytorch\\Datasets\\test\\datasets.txt", 2000)
    #
    # testdataset.ReadDataset()
    # print(testdataset.TypeNameList)
    # testdataset.BananceDataset()
    # testdataset.RewriteDataSetFile()
    # testdataset.PrintString()
