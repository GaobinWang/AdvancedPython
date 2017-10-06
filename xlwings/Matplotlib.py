#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:53:13 2017

@author: Lesile
"""

"""
Matplotlib
Using pictures.add(), it is easy to paste a Matplotlib plot as picture in Excel.
"""
###一个例子
import matplotlib.pyplot as plt
import xlwings as xw

fig = plt.figure()
plt.plot([1, 2, 3])

sht = xw.Book().sheets[0]
sht.pictures.add(fig, name='MyPlot', update=True)

