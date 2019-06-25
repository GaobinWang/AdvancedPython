# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:34:11 2019

@author: Lesile
"""
#%%
import os
import pandas as pd
import numpy as np
from datetime import timedelta #调用子模块
from datetime import datetime #调用子模块
import seaborn as sns
#%%
path = "E:\\Github\\AdvancedPython\\Pandas"
os.chdir(path)


#%%

#python数据科学生态环境的强大力量建立在NumPy与Pandas的基础之上,并通过直观的语法将基本操作转化成C语言
#在Numpy中是向量化/广播运算,在pandas中是分组运算
#虽然这些抽象功能可以简洁高效地解决许多问题,但是他们经常需要创建临时中间对象,这样就会占用大量的计算时间与内存

#python从2014年开始就引入实验性工具,让用户可以直接运行C语言的速度操作,不需要十分费力地配置中间数组,他们就是
#eval和query,都依赖于numexpr程序包

rng = np.random.RandomState(42)
x = rng.rand(10**8)
y = rng.rand(10**8)
%timeit x + y

%timeit np.fromiter((xi+yi for xi,yi in zip(x,y)),dtype = np.float, count = len(x))

###向量运算在处理复合代数式时效率比较低
mask = (x > 0.5) & (y < 0.5)
#上式等价于
tmp1 = (x > 0.5)
tmp2 = (y < 0.5)
mask = tmp1 & tmp2


import numexpr
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
np.allclose(mask, mask_numexpr)


#%%用pd.eval()实现高性能运算

nrows, ncols = 100000,10
rng = np.random.RandomState(42)
df1,df2,df3,df4 = (pd.DataFrame(rng.rand(nrows,ncols))  for i in range(4))



%timeit df1 + df2 + df3 +df4 
%timeit pd.eval('df1 + df2 + df3 +df4')



###pd.eval支持的运算
df1,df2,df3,df4,df5 = (pd.DataFrame(rng.randint(0,1000,(100,3))) for i in range(5))
#算术运算
result1 = - df1 * df2 /(df3 + df4) - df5
result2 = pd.eval('- df1 * df2 /(df3 + df4) - df5') 
np.allclose(result1,result2)
#比较运算符
result1 = (df1 < df2) & (df3 <= df4) & (df3 != df4)
result2 = pd.eval('(df1 < df2) & (df3 <= df4) & (df3 != df4)')
np.allclose(result1,result2)


###pd.eval 列间运算
df = pd.DataFrame(rng.rand(1000,3),columns = ['A','B','C'])


result1 = (df.A + df.B)/df.C - 1
result2 = df.eval('(A + B)/C - 1')
np.allclose(result1,result2)

df.eval('D = (A + B)/C - 1',inplace = True)

##pd.eval 使用局部变量
columns_mean = df.mean(1)
result1 = df['A'] + columns_mean
result2 = df.eval('A + @columns_mean')
np.allclose(result1,result2)

#%%%pd.query

result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1,result2)

result2 = df.query('A < 0.5 & B < 0.5')
np.allclose(result1,result2)

Cmean = df.C.mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean & B < @Cmean')
np.allclose(result1,result2)

"""
http://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html
"""