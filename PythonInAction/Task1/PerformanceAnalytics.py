#!/usr/bin/python
# -*- coding:utf-8 -*-

_author__ = 'Gaobin Wang'

"""
功能：计算评价策略表现的各种指标
annual_return：年化收益
max_drawdown：最大回撤
volatility：波动率
beta:计算beta
alpha:计算alpha
sharpe_ratio：计算夏普比率
"""


import os,time

# 计算年化收益率函数
def annual_return(date_line, capital_line):
    """
    :param date_line: 日期序列（list）
    :param capital_line: 账户价值序列（list）
    :return: 输出在回测期间的年化收益率
    """
    # 将数据序列合并成dataframe并按日期排序
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'date': date_line, 'capital': capital_line})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    rng = pd.period_range(df['date'].iloc[0], df['date'].iloc[-1], freq='D')
    # 计算年化收益率
    annual = 365*(df.ix[len(df.index) - 1, 'capital'] / df.ix[0, 'capital'] - 1)/len(rng)
    print('年化收益率为：%f' % annual)
    return(annual)

# 计算最大回撤函数
def max_drawdown(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return: 输出最大回撤
    """
    # 将数据序列合并为一个dataframe并按日期排序
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'date': date_line, 'capital': capital_line})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 计算当日之前的账户最大价值
    maxhere = []
    for i in range(len(capital_line)):
        temp = capital_line[0:(i+1)]
        maxheretemp = max(temp)
        maxhere.append(maxheretemp)
    df['max2here'] = maxhere

    # 计算当日的回撤
    df['dd2here'] = df['capital'] / df['max2here'] - 1

    # 计算最大回撤和结束时间
    temp = df.sort_values(by='dd2here').iloc[0][['date', 'dd2here']]
    max_dd = temp['dd2here']
    end_date = temp['date']

    # 计算开始时间
    df = df[df['date'] <= end_date]
    start_date = df.sort_values(by='capital', ascending=False).iloc[0]['date']
    print('最大回撤为：%f, 开始日期：%s, 结束日期：%s' % (max_dd, start_date, end_date))
    return(max_dd)

# 计算收益波动率的函数
def volatility(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return: 输出回测期间的收益波动率
    """
    import pandas as pd
    import numpy as np
    date_line = date_line[1:len(date_line)]
    return_line = np.diff(capital_line)/np.array(capital_line[0:(len(capital_line)-1)])
    from math import sqrt
    df = pd.DataFrame({'date': date_line, 'rtn': return_line})
    # 计算波动率
    vol = np.std(df['rtn']) * np.sqrt(250)
    print('收益波动率为：%f' % vol)
    return(vol)

# 计算贝塔的函数
def beta(date_line, capital_line, index_capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :param index_capital_line: 指数的价值序列
    :return: 输出beta值
    """
    #由净值序列计算收益率序列
    import pandas as pd
    import numpy as np
    date_line = date_line[1:len(date_line)]
    return_line = np.diff(capital_line)/np.array(capital_line[0:(len(capital_line)-1)])
    index_return_line = np.diff(index_capital_line)/np.array(index_capital_line[0:(len(index_capital_line)-1)])

    df = pd.DataFrame({'date': date_line, 'rtn': return_line, 'benchmark_rtn': index_return_line})
    # 账户收益和基准收益的协方差除以基准收益的方差
    b = df['rtn'].cov(df['benchmark_rtn']) / df['benchmark_rtn'].var()
    print('beta: %f' % b)
    return(b)

# 计算alpha的函数
def alpha(date_line, capital_line, index_capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :param index_capital_line: 指数序列
    :return: 输出alpha值
    """
    import pandas as pd
    import numpy as np
    date_line = date_line[1:len(date_line)]
    return_line = np.diff(capital_line) / np.array(capital_line[0:(len(capital_line) - 1)])
    index_return_line = np.diff(index_capital_line) / np.array(index_capital_line[0:(len(index_capital_line) - 1)])
    capital_line = capital_line[1:len(capital_line)]
    index_capital_line = index_capital_line[1:len(index_capital_line)]
    # 将数据序列合并成dataframe并按日期排序
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'date': date_line, 'capital': capital_line, 'benchmark': index_capital_line, 'rtn': return_line,
                       'benchmark_rtn': index_return_line})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    rng = pd.period_range(df['date'].iloc[0], df['date'].iloc[-1], freq='D')
    rf = 0.0284  # 无风险利率取10年期国债的到期年化收益率

    annual_stock = 365*(df.ix[len(df.index) - 1, 'capital'] / df.ix[0, 'capital'] - 1)/len(rng)  # 账户年化收益
    annual_index = 365 * (df.ix[len(df.index) - 1, 'benchmark'] / df.ix[0, 'benchmark'] - 1) / len(rng) # 基准年化收益

    beta = df['rtn'].cov(df['benchmark_rtn']) / df['benchmark_rtn'].var()  # 计算贝塔值
    a = (annual_stock - rf) - beta * (annual_index - rf)  # 计算alpha值
    print('alpha：%f' % a)
    return(a)

# 计算夏普比函数
def sharpe_ratio(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return: 输出夏普比率
    """
    # 由净值序列计算收益率序列
    import pandas as pd
    import numpy as np
    date_line = date_line[1:len(date_line)]
    return_line = np.diff(capital_line) / np.array(capital_line[0:(len(capital_line) - 1)])
    capital_line = capital_line[1:len(capital_line)]
    from math import sqrt
    # 将数据序列合并为一个dataframe并按日期排序
    df = pd.DataFrame({'date': date_line, 'capital': capital_line, 'rtn': return_line})
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    rng = pd.period_range(df['date'].iloc[0], df['date'].iloc[-1], freq='D')
    rf = 0.0284  # 无风险利率取10年期国债的到期年化收益率
    # 账户年化收益
    annual_stock = 365*(df.ix[len(df.index) - 1, 'capital'] / df.ix[0, 'capital'] - 1)/len(rng)
    # 计算收益波动率
    volatility = np.std(df['rtn']) * np.sqrt(250)
    # 计算夏普比
    sharpe = (annual_stock - rf) / volatility
    print('sharpe_ratio: %f' % sharpe)
    return(sharpe)



    






