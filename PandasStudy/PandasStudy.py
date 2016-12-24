#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
10分钟介绍pandas包的主要功能
"""
###加载相应的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

###改变工作路径
os.chdir("E:\\Github\\AdvancedPython\\PandasStudy")

###创建对象
#Series:通过一个列表创建一个Series对象
#python中的缺失值np.nan
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#DataFrame:通过numpy库中的array数据结构创建DataFrame
#python中产生日期序列的方法：pd.date_range()
#python中产生随机数的方法：np.random.randn()
dates = pd.date_range('20130101',periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index = dates, columns=list('ABCD'))
print(df)
#通过传入一个字典来创建一个DataFrame对象
dict = {
    'A':1.,
    'B':pd.Timestamp('20130101'),
    'C':pd.Series(1,index=list(range(5)),dtype='float32'),
    'D':np.array([3]*5,dtype='int32'),
    'E':pd.Categorical(["test","train","test","train","test"]),
    "F":'foo'
}
print(dict)
df2=pd.DataFrame(dict)
print(df2)
df2.dtypes


###查看数据
#查看DataFrame的前几行和后几行
df.head(3)
df.tail(3)
#DataFrame的标签
df.index
#DataFrame的列名称
df.columns
#DataFrame的数值
df.values
#DataFrame的描述
df.describe()
#DataFrame的转置
df.T
#按照行或者列的索引进行排序
df.sort_index(axis=0,ascending = False) #按照列索引降序排列
df.sort_index(axis=1,ascending = False) #按照行索引降序排列
#按照某一行或某一列的数值进行排序
df.sort_values(by='B')

#DataFrame的子集选择
#获取某一列
df.A
df['A']
#获取某几行
df[0:3] #获取第一行到第四行
df['20010101':'20010130'] #获取两个索引之间的数据
#通过标签来获取数据
a=df.loc[dates[0]]
print(a)
df.loc[:,['A','B']]
df.loc['20130102':'20130104',['A','B']]
df.loc['20130102',['A','B']]
df.loc[dates[0],'A']
df.at[dates[0],'A'] #与前一种方法类似，但是速度更快
#通过位置来获取数据
df.iloc[3]
df.iloc[3:5,0:2]
df.iloc[[1,2,4],[0,2]] #相对位置的列表
df.iloc[1:3,:]
df.iloc[:,1:3]
df.iloc[1,1] #获取一个具体的数值
df.iat[1,1] #与之前的方法一样，但是速度更快
#通过Boolean Indexing
df[df.A > 0]
df[df > 0]
df2 = df.copy()
df2['E'] = ['one','one','two','three','four','four']
df2
df2[df2['E'].isin(['two','four'])]

###改变DataFrame
#给DataFrame增加一列
s1 = pd.Series([1,2,3,4,5,6], index = pd.date_range('20130102',periods = 6))
df['F']=s1
#该表DataFrame某个位置的数值
df.at[dates[0],'A'] = 0  #通过标签确定位置
df.iat[0,0] = 0 #通过相对位置来确定位置
df.loc[:,'D'] = np.array([5]*len(df))
#通过where来改变某个位置的数值
df2 = df.copy()
df2[df2 > 0] = -df2
df11=df

###缺失值的处理
#reindex能够改变、增加、删除一个索引
df1 = df.reindex(index = dates[0:4],columns = list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
#去除掉任何有缺失值的行
df1.dropna(how='any')
#填补缺失值
df1.fillna(value=5)
#返回是否缺失的逻辑变量
pd.isnull(df1)


###操作运算符
#计算每一列的均值
df.mean()
#计算每一行的均值
df.mean(1)

#当维度不一样的时候，DataFrame会自动的填补
#shift 和 sub函数
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
df.sub(s, axis='index')

#对某一行或者某一列应用某个函数
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

#统计函数
s = pd.Series(np.random.randint(0,7,size=10))
s.value_counts()

#字符串方法
s = pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
s.str.lower()

###Merge
#concat
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)
#join
left = pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
right = pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})
pd.merge(left,right,on='key')
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar','foo'], 'rval': [1,4, 5]})
pd.merge(left,right,on='key')
#append
df = pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s,ignore_index = True)

###分组
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                           'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                    'C' : np.random.randn(8),
                'D' : np.random.randn(8)})

df.groupby('A').sum()
df.groupby(['A','B']).sum()


###reshaping
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                     ['one', 'two', 'one', 'two',
                      'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
stacked = df2.stack()
stacked.unstack()
stacked.unstack(1)
stacked.unstack(0)
#Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                       'B' : ['A', 'B', 'C'] * 4,
                    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                    'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


###时间序列
rng = pd.date_range('1/1/2012',periods=100,freq='S')
help(pd.date_range)
ts = pd.Series(np.random.randint(0,500,len(rng)),index = rng)
ts.resample('5Min').sum()
#改变时区
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')
#改变时间点
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()
#分类变量
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df.sort_values(by="grade")  #按照分类变量进行排序
df.groupby("grade").size()





















###Plot
#Plot1
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts=ts.cumsum()
plt.figure()
ts.plot()
plt.show()
#Plot2
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc = 'best')
plt.show()

#http://pandas.pydata.org/pandas-docs/stable/10min.html#min

###Getting Data In/Out
#csv
df.to_csv('foo.csv')
data = pd.read_csv('foo.csv')
#Excel
df.to_excel("foo.xlsx",sheet_name='Sheet1')
data = pd.read_excel('foo.xlsx','Sheet1',index_col = None, na_values=['NA'])

