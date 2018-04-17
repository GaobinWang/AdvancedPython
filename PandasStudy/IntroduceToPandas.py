# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 04:13:59 2018

@author: Lesile

Ref:https://zhuanlan.zhihu.com/p/25184830
"""

###创建数据
#1，引入pandas
import numpy as np
import pandas as pd 

#2,创建数据集
data2 = pd.DataFrame(np.random.rand(6,4),columns = list('ABCD'))

#创建一个时间索引
dates = pd.date_range('20170601',periods=6)
df = pd.DataFrame(np.random.rand(6,4),index = dates,columns = list('ABCD'))

#用字典来创建数据
df2 = pd.DataFrame({'A':np.random.rand(3)})
df2 = pd.DataFrame({'A':np.random.rand(3)},index = pd.date_range('20110101',periods=3))

#另一种用字典创建数据的方法
df3 = pd.DataFrame({'A':np.random.rand(3),'B':np.random.rand(3)})


###筛选数据
df = pd.DataFrame(np.random.randn(6,4),columns = list('ABCD'))

#简单筛选
df1 = df[df.A>0]
#多条件筛选
df2 = df[(df.A>0) & (df.B<0)] #and
df3 = df[(df.A>0) | (df.B<0)] #or
#限定我们需要的列之后筛选
df4 = df[['A','B']][(df.A>0) & (df.B<0)]
#其他筛选方法
temp = np.array(df.A)[1:3]
index = df.A.isin(temp)
df5 = df[index]


###增加和删除列
#原始数据
df = pd.DataFrame(np.random.randn(6,4),index = pd.date_range('20180101',periods = 6),columns=list('ABCD'))
#插入一列
df['E'] = pd.Series(np.random.randn(6),index=df.index)
#在某一固定位置插入yilie
df.insert(1,'AA',np.random.randn(6))
#永久的删除一列
del df['AA']
#暂时删除一列
df1 = df.drop(['A','B'],axis=1)

###数据分组
data = {'ID':np.random.randint(low=1,high=6,size=1000),
        'month':np.random.randint(low=1,high=13,size=1000),
        'buy':np.random.random(1000),
        'sell':np.random.random(1000)}
df = pd.DataFrame(data,columns = data.keys())

#通过groupby实现数据分组
group_ID = df.groupby('ID')
group_ID.first() #分组后每一组的第一行数据
group_ID.last() #分组后每一组的最后一行数据

group_ID2 = df.groupby(['ID','month'])
group_ID2.first()
group_ID2.first().head() #展现前5行数据

#
group_ID2.size() #分组计算每组的数据量
group_ID2.describe() #分组进行描述性统计分析

#aggregate函数的使用
df1 = df.groupby(['ID','month']).aggregate(np.sum)

#agg的使用
df2 = df.groupby(['ID','month']).agg([np.mean,np.std,np.sum])
#df3 = df.groupby(['ID','month']).agg({'s':'max'})为何重命名功能不能正常使用呢


#agg的用法
df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],
                  index=pd.date_range('1/1/2000', periods=10))
df.iloc[3:7] = np.nan
df.agg(['sum', 'min'])
df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})


###缺失值处理
#创建数据
df = pd.DataFrame(np.random.randn(4,3),index=list('abcd'),columns=['one','two','three'])
df.ix[1,:-1] = np.nan
df.ix[1:-1,2] = np.nan

#用固定值替代缺失值
df.fillna(0) #用数值0替代
df.fillna('missing') #用字符串替代

#用临近的值替代
df.fillna(method='pad') #用前面的数值替代
df.fillna(method='bfill')
df.fillna(method='bfill',limit=1)#限制每列可以替代的数量

#用统计值替换缺失值
df.fillna(df.mean())
df.fillna(df.mean()['one':'two']) #还可以选择哪一列进行缺失值的处理

#用插补的方法填补缺失值
df.interpolate()

#删除缺失值
df.dropna(axis=0) #按行删除

#值替换
#具体数值的替换
ser = pd.Series([0,1,2,3,4])
print(ser.replace(0,8))
#列表到列表的替换
ser.replace([0,1,2,3,4],[4,3,2,1,0])
#使用字典映射，如将1替换为11，将2替换为12
ser.replace({1:11,2:12})

#以上方法同样适用于DataFrame对象
df = pd.DataFrame({'a':[0,1,2,3],'b':[2,6,7,3]})
df.replace(2,20)
df['a'].replace([0,1,2,3],[5,4,3,2])

