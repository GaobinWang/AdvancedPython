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
np.empty((2,3),dtype=np.int16) #随机给

#创建一个序列
np.arange(10,30,5) #从10到30，间隔为5的序列
np.arange(0,2,0.3)
np.linspace(0,2,9) #9 numbers from 0 to 2
from numpy import pi
x = np.linspace(0,2*pi,100)
f = np.sin(x)

#生成随机数
np.random.rand(3,2) #生成一个3*2维的ndarray,元素服从0-1间均匀分布
np.random.randn(100) #生成一个1*100维的ndarray,元素服从标准正太分布
sigma = 2.5
mu = 2
sigma*np.random.randn(100)+mu #生成一个1*100维的ndarray,元素服从均值为2，标准差为2.5的正太分布
np.random.randint(0,2,20) #从0:(2-1)中生成20个随机整数
np.random.randn(100)

#基本运算符
a=[20,30,40,50]
b=range(4)
#c=a-b  这样会报错
a = np.array(a)
b = np.array(b)
c = a- b #支持逐个元素间的运算

#逐个元素相乘
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
A*B

#矩阵的乘法
A.dot(B)
np.dot(A,B)

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
x = np.array([1,2,4,7,0])
print(x)
print(np.diff(x))
np.diff(x) #一阶差分，默认n=1
np.diff(x,n=2) #二阶差分
x = np.array([[1,3,6,10],[0,5,6,8]])
np.diff(x)  #按行差分
np.diff(x,axis=0) #按列差分

#cumsum
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
#一维ndarray的索引与list相同
def f(x,y):
    return(10*x + y)
b = np.fromfunction(f,(5,4),dtype = int)
#?fromfunction
b[2,3]
b[0:5,1]
b[-1] #最后一列


#其他数据操作

































