import threading
import time


class A(threading.Thread):
    def __init__(self,name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self) -> None:
        while True:
            time.sleep(2)
            print("A class : name = {}".format(self.name))

    def setName1(self, name: str) -> None:
        print("A : set name = {}".format(name))
        self.name = name

class B(threading.Thread):
    def __init__(self,age):
        threading.Thread.__init__(self)
        self.age = age

    def run(self) -> None:
        while True:
            time.sleep(2)
            print("B class : age = {}".format(self.age))

    def setAge1(self,age)-> None:
        print("B : set age = {}".format(age))
        self.age = age

class C(A,B):
    def __init__(self,name,age):
        A.__init__(self,name=name)
        B.__init__(self,age=age)

    def setName1(self, name: str) -> None:
        print("C : set name = {}".format(name))
        A.name = name

    def setAge1(self,age) -> None:
        print("C : set age = {}".format(age))
        B.age = age

if __name__ == '__main__':
    c = C("kkk", 19)
    c.setAge1("mrmzy")
    c.setAge1(40)
