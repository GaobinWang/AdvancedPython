#!/usr/bin/python
# -*- coding:utf-8 -*-
_author__ = 'Gaobin Wang'
"""
功能：计算托管产品的评价指标
输入：副本产品净值.xlsx
输出：托管产品汇总.xlsx
"""

###导入用到的库
import pandas as pd
import  numpy as np
import os,time
from WindPy  import w
import PerformanceAnalytics


###设置工作路径
path = os.getcwd()
os.chdir(path)
os.chdir("E:\\Github\\AdvancedPython\\PythonInAction\\Task1")

###读取数据
#托管产品数据
AllNetValueData = pd.read_excel('副本产品净值.xlsx','SQL Results',index_col = None, na_values=['NA'])
#沪深300指数数据
HS300Data = pd.read_csv('沪深300指数.csv')
"""
从Wind中提取沪深300指数数据
w.start()
HS300 = w.wsd("000300.SH","close",'20100101','20161223')
HS300Data = HS300.Data
HS300Data = HS300Data[0]
HS300Times = HS300.Times
data = pd.DataFrame({'date_line':HS300Times,'index_capital_line':HS300Data})
data.to_csv('沪深300指数.csv',index=None)
"""

###计算产品评价指标
AllProduct = np.unique(AllNetValueData['VC_CPMC'])
result = pd.DataFrame(AllProduct,columns = ['产品名称'],index = AllProduct)
#数据清洗(处理日期数据)
for i in range(len(HS300Data.index)):
    print(i)
    temp = HS300Data.iloc[i,0]
    temp = temp[0:10]
    temp= time.strptime(str(temp), "%Y-%m-%d")
    HS300Data.iloc[i, 0] = time.strftime('%Y%m%d',temp)

for i in range(len(AllNetValueData.index)):
    print(i)
    temp = AllNetValueData.iloc[i, 0]
    temp = time.strptime(str(temp), "%Y%m%d")
    AllNetValueData.iloc[i, 0] = time.strftime('%Y%m%d',temp)
Product = AllProduct[0]
for Product in  AllProduct:
    print("########",Product,"#######")
    #读取产品Product的数据，并按照日期进行排序
    NetValueData = AllNetValueData[AllNetValueData['VC_CPMC'] == Product]
    NetValueData.sort_values(by='TRADE_DATE', inplace=True)
    NetValueData.reset_index(drop=True, inplace=True)
    NetValueDataThisYear = NetValueData[NetValueData['TRADE_DATE'] >= '20160101']

    #处理数据
    ProductData = NetValueData.loc[:,['TRADE_DATE','PER_NV']]
    ProductData.columns = ['date_line','capital_line']
    HS300Data.columns = ['date_line','index_capital_line']
    Data = pd.merge(ProductData,HS300Data, on = 'date_line')  #非交易日产品净值为何会有变动
    DataThisYear = Data[Data['date_line'] >= '20160101']
    HS300ThisYear = HS300Data[HS300Data['date_line'] >= '20160101']
    
    
    #1,产品代码
    result.loc[Product,'产品代码']=np.unique(NetValueData['VC_CPDM'])[0]

    #2,产品今年以来收益率%
    StartValue = NetValueDataThisYear.iloc[0,7]
    EndValue = NetValueDataThisYear.iloc[-1,7]
    value = (EndValue/StartValue - 1)*100
    result.loc[Product,'产品今年以来收益率%'] = round(value,4)

    #3，沪深300今年以来收益%
    StartValue = HS300ThisYear.iloc[0,1]
    EndValue = HS300ThisYear.iloc[-1,1]
    value = (EndValue/StartValue - 1)*100
    result.loc[Product,'沪深300今年以来收益%'] = round(value,4)
    
    #4，产品今年最大回撤%
    date_line = list(DataThisYear.date_line)
    capital_line = list(DataThisYear.capital_line)
    value = PerformanceAnalytics.max_drawdown(date_line,capital_line)
    value = value*100
    result.loc[Product,'产品今年最大回撤%'] = round(value,4)

    #5,沪深300今年最大回撤%
    date_line = list(HS300ThisYear.date_line)
    capital_line = list(HS300ThisYear.index_capital_line)
    value = PerformanceAnalytics.max_drawdown(date_line, capital_line)
    value = value * 100
    result.loc[Product, '沪深300今年最大回撤%'] = round(value, 4)

    #6,产品今年波动率%
    date_line = list(DataThisYear.date_line)
    capital_line = list(DataThisYear.capital_line)
    value = PerformanceAnalytics.volatility(date_line, capital_line)
    value = value * 100
    result.loc[Product,'产品今年波动率%'] = round(value, 4)

    #7,沪深300今年波动率%
    date_line = list(HS300ThisYear.date_line)
    capital_line = list(HS300ThisYear.index_capital_line)
    value = PerformanceAnalytics.volatility(date_line, capital_line)
    value = value * 100
    result.loc[Product,'沪深300今年波动率%'] = round(value, 4)

    #8,产品sharp值
    date_line = list(Data.date_line)
    capital_line = list(Data.capital_line)
    value = PerformanceAnalytics.sharpe_ratio(date_line, capital_line)
    result.loc[Product,'产品sharp值'] = round(value, 4)

    #9，沪深300sharp值
    date_line = list(HS300ThisYear.date_line)
    capital_line = list(HS300ThisYear.index_capital_line)
    value = PerformanceAnalytics.sharpe_ratio(date_line, capital_line)
    result.loc[Product,'沪深300sharp值'] = round(value, 4)

    #10,beta alpha
    date_line = list(Data.date_line)
    capital_line = list(Data.capital_line)
    index_capital_line = list(Data.index_capital_line)
    value1 = PerformanceAnalytics.beta(date_line, capital_line,index_capital_line)
    value2 = PerformanceAnalytics.alpha(date_line, capital_line, index_capital_line)
    result.loc[Product,'beta'] = round(value1,4)
    result.loc[Product, 'alpha'] = round(value2, 4)

    #11,当前交易日()
    result.loc[Product,'当前交易日']=time.strftime('%Y%m%d',time.localtime(time.time()))

    #12,账户统计日(TRADE_DATE)
    temp = NetValueData.iloc[-1,0]
    temp = time.strptime(str(temp), "%Y%m%d")
    result.loc[Product,'账户统计日'] = time.strftime('%Y%m%d',temp)

    #13,产品规模(ALL_NETV_MKV)
    result.loc[Product,'产品规模'] = NetValueData.iloc[-1,5]

    #14,统计日单位净值(PER_NV)
    result.loc[Product,'统计日单位净值'] = NetValueData.iloc[-1,7]

    #15,产品份额(SHARES)
    result.loc[Product,'产品份额'] = NetValueData.iloc[-1,9]

    #16,成立以来收益率
    StartValue = NetValueData.iloc[0, 7]
    EndValue = NetValueData.iloc[-1, 7]
    value = (EndValue / StartValue - 1) * 100
    result.loc[Product,'成立以来收益率%'] = round(value,4)

    #17,今年以来收益率
    StartValue = NetValueDataThisYear.iloc[0,7]
    EndValue = NetValueDataThisYear.iloc[-1,7]
    value = (EndValue / StartValue - 1) * 100
    result.loc[Product,'今年以来收益率%'] = round(value,4)

    #18,账户起始日
    temp = NetValueData.iloc[0,0]
    temp = time.strptime(str(temp), "%Y%m%d")
    result.loc[Product,'账户起始日'] = time.strftime('%Y%m%d',temp)

    #19,起始日单位净值
    result.loc[Product,'起始日单位净值'] = NetValueData.iloc[0,7]

print(result)
result.to_excel("托管产品汇总.xlsx",sheet_name='Sheet1')
    
    
    
    






