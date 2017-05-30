#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
numpy快速入门
参考资料：https://docs.scipy.org/doc/numpy/user/quickstart.html
简介：
1，numpy中的主要数据类型是多维数组(数组中每一个元素的数据类型相同)，在numpy中维度成为axis,
维度的数值称为rank；
2，numpy中数组的类型被称为ndarray(简称为array),需要注意的是array.array和numpy.array是不同的，
array.array仅仅具有一个维度，且提供的函数更少；
3，ndarray的几个重要属性：
ndarray.ndim:维度
ndarray.shape:数组的各个维度的长度
ndarray.size:数组中元素的总的个数（nrow(data)*ncol(data)）
ndarray.dtype:数组的数据类型
ndarray.itemsize:数组中每个元素所占物理空间的大小
ndarray.data:ndarray中的数据，也可以通过index的方式获取
"""

###加载相应的库
import numpy as np

#例子
a = np.arange(15).reshape(3,5)
print(a)
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)

#创建ndarray对象(通过传入list或者tuple)
a = [2,3,4]
a = np.array(a)
b = np.array([(1.5,2,3),(4,5,6)])
print(b)
c = np.array([[1,2],[3,4]],dtype = complex) #在创建ndarray时指定数据类型
print(c)

#创建指定维度的ndarray
np.zeros((3,4)) #均为0
np.ones((2,3,4),dtype=np.int16) #均为1
np.empty((2,3),dtype=np.int16) #
# empty会创建一个没有任何具体值的数组，
#认为np.empty会返回全0数组的想法是不安全的。很多情况下，它返回的都是一些未初始化的垃圾值
#empty的用途？

#创建一个序列
#np.arange是Python内置函数range的数组版
np.arange(10,30,5) #从10到30，间隔为5的序列
np.arange(0,2,0.3)
np.linspace(0,2,9) #9 numbers from 0 to 2
from numpy import pi
x = np.linspace(0,2*pi,100)
f = np.sin(x)

#ndarray数据类型的转换
data = np.arange(10)
data.dtype
data = data.astype(np.float)  #将整型转化为浮点型
data.dtype
data = np.array([1.123,2.234,3.456])
data = data.astype(np.int8) #将浮点型转化为整型，则小数部分将会被截断
data

#生成随机数
#numpy.random模块对Python内置的random进行了补充，增加了一些用于高效生成多种概率分布的样本值函数
np.random.rand(3,2) #生成一个3*2维的ndarray,元素服从0-1间均匀分布
from numpy.random import rand
rand(3,2)
np.random.randn(100) #生成一个1*100维的ndarray,元素服从标准正太分布
from numpy.random import randn
randn(5)
sigma = 2.5
mu = 2
sigma*np.random.randn(100)+mu #生成一个1*100维的ndarray,元素服从均值为2，标准差为2.5的正太分布
np.random.randint(0,2,20) #从0:(2-1)中生成20个随机整数
np.random.randn(100)

data = np.random.normal(loc = 0, scale = 1,size = (4,4)) #4*4的标准正态分布
#下面的测试显示了如果要生成大量样本值，numpy.random比Python内置的random模块快很多
from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0,1)  for _ in range(N)]
%timeit samples = np.random.normal(N)

#%timeit的用法



#基本运算符
a=[20,30,40,50]
b=range(4)
#c=a-b  这样会报错
a = np.array(a)
b = np.array(b)
c = a- b #支持逐个元素间的运算
a = np.array(range(10))
b = np.array(range(5))
#a - b #这种写法是错误的，Python和R在这一点上右很大不同

#数组和标量之间的运算
data = np.arange(10).reshape(2,5)
data2 = 1/data
data3 = data ** 2

#逐个元素相乘
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
A*B

#矩阵的乘法
A.dot(B)
np.dot(A,B)
np.dot(A.T,B)

#一些运算符，比如： +=  *= 的作用是修改现存的数据，而非创建新的数据
a = np.ones((2,3),dtype=int)
b = np.random.random((2,3))
a *= 3
b += a  #float = float + int
#a += b #错误

#当两个不同数据类型的array做运算时，返回的结果是更general的数据类型

#指数运算
a=np.arange(3)
b=np.exp(a)
c = a * b
a.dtype
b.dtype
c.dtype


#统计函数
a = np.ones((2,3))
a.max()
a.min()
a.sum()

#以上函数默认情况下会把ndarray当成一个list来处理，而不管他的维度。
#然而，我们也可以通过声明axis参数来指明运算的方向
b = np.arange(12).reshape(3,4)
print(b)
b.sum(axis=0) #sum of each column
b.min(axis=1) #min of each row
b.cumsum(axis=1) #cumulative sum along each row

#常用的数学函数
a = np.arange(10)
np.exp(a) #指数运算
np.sqrt(a) #
np.add(a,a) #相加

#diff函数的用法
#diff为做差分的函数diff(a,n,axis),其中a为做差分的对象，n为几阶差分，axis为按照那个方向进行差分
x = np.array([1,2,4,7,0])
print(x)
print(np.diff(x))
np.diff(x) #一阶差分，默认n=1
np.diff(x,n=2) #二阶差分
x = np.array([[1,3,6,10],[0,5,6,8]])
np.diff(x)  #按行差分
np.diff(x,axis=0) #按列差分

#cumsum  (cumprod的用法类似)
a = np.array([[1,2,3], [4,5,6]])
np.cumsum(a)
np.cumsum(a, dtype=float)     # specifies type of output value(s)
np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows

#round函数

#其他函数
a =np.arange(10)
a.std()
a.var()


#ndarray的索引
#一维ndarry的索引与列表的索引类似
data = np.arange(10)
data
data[5]
data[5:8]
data[5:8] = 12  #跟列表重要的区别在于，数组切片是原始数据的视图，这意味着数据不会被复制，视图上的任何修改都会直接反映到源数据上

data2 = data[5:8]
data2[1] = 12345
data
#如果坚持要将数据复制来复制去的话可能产生何等的性能和内存问题
#如果要想得到ndarray的切片的一份副本而非视图，就需要显式的进行复制操作
data3 = data[2:5].copy()

#对于高维数组，各索引位置上的元素不再是标量而是一个一维数组


b = np.arange(30).reshape(5,6)
b[2,3]
b[0:5,1]
b[:,-1] #最后一列

#？
def f(x,y):
    return(10*x + y)
b = np.fromfunction(f,(5,4),dtype = int)
#?fromfunction

































