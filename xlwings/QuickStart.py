#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:31:16 2017

@author: Lesile
"""

"""
###xlwings简介
xlwings - Make Excel Fly!
xlwings is a BSD-licensed Python library that makes it easy to call Python from Excel and vice versa:

Scripting: Automate/interact with Excel from Python using a syntax close to VBA.
Macros: Replace VBA macros with clean and powerful Python code.
UDFs: Write User Defined Functions (UDFs) in Python (Windows only).
Numpy arrays and Pandas Series/DataFrames are fully supported. 
xlwings-powered workbooks are easy to distribute and work on Windows and Mac.
"""

###1. Scripting: Automate/interact with Excel from Python
#Establish a connection to a workbook:
import xlwings as xw
wb = xw.Book()  # this will create a new workbook
wb = xw.Book('FileName.xlsx')  # connect to an existing file in the current working directory
wb = xw.Book(r'C:\path\to\file.xlsx')  # on Windows: use raw strings to escape backslashes    

#If you have the same file open in two instances of Excel, you need to fully qualify it and include the app instance:
xw.apps[0].books['FileName.xlsx']

#Instantiate a sheet object:
sht = wb.sheets['Sheet1']

#Reading/writing values to/from ranges is as easy as:
sht.range('A1').value = 'Foo 1'
sht.range('A1').value

#There are many convenience features available, e.g. Range expanding:
sht.range('A1').value = [['Foo 1', 'Foo 2', 'Foo 3'], [10.0, 20.0, 30.0]]
sht.range('A1').expand().value

#Powerful converters handle most data types of interest, including Numpy arrays and Pandas DataFrames in both directions:
import pandas as pd
df = pd.DataFrame([[1,2], [3,4]], columns=['a', 'b'])
sht.range('A1').value = df
sht.range('A1').options(pd.DataFrame, expand='table').value

#Matplotlib figures can be shown as pictures in Excel:
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot([1, 2, 3, 4, 5])
sht.pictures.add(fig, name='MyPlot', update=True)


#Shortcut for the active sheet: xw.Range

#If you want to quickly talk to the active sheet in the active workbook, you don’t need instantiate a workbook and sheet object, but can simply do:
import xlwings xw
xw.Range('A1').value = 'Foo'
xw.Range('A1').value
#Note: You should only use xw.Range when interacting with Excel. In scripts, you should always go via book and sheet objects as shown above.

###2. Macros: Call Python from Excel
#You can call Python functions from VBA using the RunPython function:
Sub HelloWorld()
    RunPython ("import hello; hello.world()")
End Sub

#Per default, RunPython expects hello.py in the same directory as the Excel file. Refer to the calling Excel book by using xw.Book.caller:

# hello.py
import numpy as np
import xlwings as xw

def world():
    wb = xw.Book.caller()
    wb.sheets[0].range('A1').value = 'Hello World!'
#To make this run, you’ll need to have the xlwings add-in installed. The easiest way to get everything 
#set up is to use the xlwings command line client from either a command prompt on Windows or a terminal on Mac: xlwings quickstart myproject.
#For details about the addin, see Add-in.

#3. UDFs: User Defined Functions (Windows only)
Writing a UDF in Python is as easy as:

import xlwings as xw

@xw.func
def hello(name):
    return 'Hello {0}'.format(name)
Converters can be used with UDFs, too. Again a Pandas DataFrame example:

import xlwings as xw
import pandas as pd

@xw.func
@xw.arg('x', pd.DataFrame)
def correl2(x):
    # x arrives as DataFrame
    return x.corr()
#Import this function into Excel by clicking the import button of the xlwings add-in: For further details, see VBA: User Defined Functions (UDFs).