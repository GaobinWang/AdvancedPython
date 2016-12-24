#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
此脚本实现的功能为对托管的产品进行信息汇总
输入：副本产品净值.xlsx
输出：托管产品汇总.xlsx
"""

###导入用到的库
import pandas as pd
from pandas import Series
import  numpy as np
import matplotlib.pyplot as plt
import os
import time
from WindPy  import w
path = os.getcwd()
os.chdir(path)

###定义用到的函数
def theround(value,n=4):
    """
    theround函数实现的功能是对value(浮点型数据)保留n位有效数字
    """
    return(int(value*10**n)/10**n)

def max_drawdown(netvalue):
    """
    max_drawdown函数的主要功能是计算序列的最大回撤
    """
    
    maxhere = [] #当前时点的最大值
    drawdown = [] #当前时点的回撤
    for i in range(len(netvalue)):
        temp = netvalue[0:(i+1)]
        maxheretemp = max(temp)
        drawdowntemp = netvalue[i]/maxheretemp - 1
        drawdowntemp = drawdowntemp*(-1)
        maxhere.append(maxheretemp)
        drawdown.append(drawdowntemp)  
    maxdrawdown = max(drawdown)
    maxdrawdown = maxdrawdown*100
    maxdrawdown = theround(maxdrawdown)
    return(maxdrawdown)
        

# 计算收益波动率的函数
def volatility(netvalue):
    """
    :param netvalue: 净值序列(list类型)
    :return: 输出回测期间的收益波动率
    """
    rtn = []
    for i in range((len(netvalue) - 1)):
        temp1 = netvalue[i]
        temp2 = netvalue[(i+1)]
        rtntemp = temp2/temp1 - 1
        rtn.append(rtntemp)
        
    from math import sqrt
    # 计算波动率
    vol = np.std(rtn)* sqrt(250)
    vol = vol*100
    vol = theround(vol)
    return(vol)
    
# 计算贝塔的函数
def beta(netvalue1, netvalue2):
    """
    :param netvalue1: 产品的净值序列
    :param netvalue2: 指数的净值序列
    :return: 输出beta值
    """
    return_line = []
    indexreturn_line = []
    for i in range((len(netvalue1))):
        return_line.append(netvalue1[(i+1)]/netvalue1[i] - 1)
        indexreturn_line.append(netvalue2[(i+1)]/netvalue2[i] - 1)
    df = pd.DataFrame({ 'rtn': return_line, 'benchmark_rtn': indexreturn_line})
    # 账户收益和基准收益的协方差除以基准收益的方差
    b = df['rtn'].cov(df['benchmark_rtn']) / df['benchmark_rtn'].var()
    return(b)

 
    

#从Wind中读取沪深300指数数据
w.start()
HS300 = w.wsd("000300.SH","close",'20160101','20161223')
HS300Data = HS300.Data
HS300Data = HS300Data[0]
#HS300Data = pd.DataFrame(HS300.Data)

#print(max_drawdown(HS300Data))


#HS300Close = HS300.Data
#print(HS300Close)

#print(HS300)

#读入数据
AllNetValueData = pd.read_excel('副本产品净值.xlsx','SQL Results',index_col = None, na_values=['NA'])
AllProduct = np.unique(AllNetValueData['VC_CPMC'])
print(AllProduct)

result = pd.DataFrame(AllProduct,columns = ['产品名称'],index = AllProduct)
print(result)
for Product in  AllProduct:
    #print(Product)
    NetValueData = AllNetValueData[AllNetValueData['VC_CPMC'] == Product]
    NetValueDataThisYear = NetValueData[NetValueData['TRADE_DATE'] > 20160100]
    n = len(NetValueData.index)
    n1 = len(NetValueDataThisYear.index)
    
    #产品代码
    result.loc[Product,'产品代码']=np.unique(NetValueData['VC_CPDM'])[0]

    #产品今年以来收益率%
    StartValue = NetValueDataThisYear.iloc[0,7]
    EndValue = NetValueDataThisYear.iloc[(n1-1),7]
    value = (EndValue/StartValue - 1)*100
    result.loc[Product,'产品今年以来收益率%'] = theround(value)

    #沪深300今年以来收益%
    StartValue = HS300Data[0]
    EndValue = HS300Data[(len(HS300Data) - 1)]
    value = (EndValue/StartValue - 1)*100
    result.loc[Product,'沪深300今年以来收益%'] = theround(value)
    
    
    #产品今年最大回撤%
    netvalue = NetValueDataThisYear.iloc[:,7]
    netvalue = list(netvalue)
    result.loc[Product,'产品今年最大回撤%'] = max_drawdown(netvalue)

    #沪深300今年最大回撤%
    netvalue = HS300Data
    netvalue = list(netvalue)
    result.loc[Product,'沪深300今年最大回撤%'] = max_drawdown(netvalue)

    #产品今年波动率%
    netvalue = NetValueDataThisYear.iloc[:,7]
    netvalue = list(netvalue)
    result.loc[Product,'产品今年波动率%'] = volatility(netvalue)

    #沪深300今年波动率%
    netvalue = HS300Data
    netvalue = list(netvalue)
    result.loc[Product,'沪深300今年波动率%'] = volatility(netvalue)

    #产品sharp值
    rf = 2.84  # 无风险利率取10年期国债的到期年化收益率
    rtn = result.loc[Product,'产品今年以来收益率%']*250/len(netvalue)
    result.loc[Product,'产品sharp值'] = (rtn-rf)/result.loc[Product,'产品今年波动率%']
    

    #沪深300sharp值
    rf = 2.84  # 无风险利率取10年期国债的到期年化收益率
    rtn = result.loc[Product,'沪深300今年以来收益%']*250/len(netvalue)
    result.loc[Product,'沪深300sharp值'] = (rtn-rf)/result.loc[Product,'产品今年波动率%']

    #beta
    netvalue1 = NetValueDataThisYear.iloc[:,7]
    netvalue1 = list(netvalue1)
    netvalue2 = HS300Data
    netvalue2 = list(netvalue2)
    #result.loc[Product,'沪深300今年波动率%'] = beta(netvalue1,netvalue2)

    #当前交易日()
    result.loc[Product,'当前交易日']=time.strftime('%Y%m%d',time.localtime(time.time()))

    #账户统计日(TRADE_DATE)
    temp = NetValueData.iloc[(n-1),0]
    temp = time.strptime(str(temp), "%Y%m%d")
    result.loc[Product,'账户统计日'] = time.strftime('%Y%m%d',temp)

    #产品规模(ALL_NETV_MKV)
    result.loc[Product,'产品规模'] = NetValueData.iloc[(n-1),5]

    #统计日单位净值(PER_NV)
    result.loc[Product,'统计日单位净值'] = NetValueData.iloc[(n-1),7]

    #产品份额(SHARES)
    result.loc[Product,'产品份额'] = NetValueData.iloc[(n-1),9]

    #成立以来收益率
    StartValue = NetValueData.iloc[0,7]
    EndValue = NetValueData.iloc[(n-1),7]
    result.loc[Product,'成立以来收益率%'] = (EndValue/StartValue - 1)*100
    

    #今年以来收益率
    StartValue = NetValueDataThisYear.iloc[0,7]
    EndValue = NetValueDataThisYear.iloc[(n1-1),7]
    result.loc[Product,'今年以来收益率%'] = (EndValue/StartValue - 1)*100

    #账户起始日
    temp = NetValueData.iloc[0,0]
    temp = time.strptime(str(temp), "%Y%m%d")
    result.loc[Product,'账户起始日'] = time.strftime('%Y%m%d',temp)
    #起始日单位净值
    result.loc[Product,'起始日单位净值'] = NetValueData.iloc[0,7]

print(result)
result.to_excel("托管产品汇总.xlsx",sheet_name='Sheet1')
    
    
    
    






