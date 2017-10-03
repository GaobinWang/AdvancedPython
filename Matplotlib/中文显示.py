# -*- coding: utf-8 -*-

'''
matplotlib中因为没有中文字体，所以无法显示中文。
针对matplotlib显示中文的办法，只需添加下面三行代码：
'''
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

#coding:utf-8
import matplotlib.pyplot as plt
plt.plot((1,2,3),(4,3,-1))
plt.xlabel(u'横坐标')
plt.ylabel(u'纵坐标')
plt.show()