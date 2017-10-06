#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:44:04 2017

@author: Lesile
"""

"""
###用pandas处理大数据———减少90%内存消耗的小贴士
网址:https://uqer.datayes.com/community/share/5993c264570651010a2e55b0
"""
import os
import pandas as pd

path = "E:\Github\AdvancedPython\Pandas"
os.chdir(path)

gl = pd.read_csv('game_logs.csv')
gl.head()
