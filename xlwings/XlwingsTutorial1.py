#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:00:23 2017

@author: Lesile
"""

"""
xlwings教程

xlwings的特色:
    xlwings能够非常方便的读写Excel文件中的数据，并且能够进行单元格格式的修改
    可以和matplotlib以及pandas无缝连接
    可以调用Excel文件中VBA写好的程序，也可以让VBA调用用Python写的程序。
    开源免费，一直在更新
"""

"""
xlwings教程1
"""


###1 打开已保存的Excel文档
# 导入xlwings模块，打开Excel程序，默认设置：程序可见，只打开不新建工作薄，屏幕更新关闭
import xlwings as xw
app=xw.App(visible=True,add_book=False)
app.display_alerts=False
app.screen_updating=False
# 文件位置：filepath，打开test文档，然后保存，关闭，结束程序
filepath = 'E:\\Github\\AdvancedPython\\xlwings\\new_book_2.xlsx'
wb=app.books.open(filepath)
wb.save()
wb.close()
app.quit()

###2 新建Excel文档，命名为test.xlsx，并保存在D盘。
import xlwings as xw
app=xw.App(visible=True,add_book=False)
wb=app.books.add()
wb.save(r'E:\\Github\\AdvancedPython\\xlwings\\test.xlsx')
wb.close()
app.quit()


###3 在单元格输入值
# 新建test.xlsx，在sheet1的第一个单元格输入 “人生” ，然后保存关闭，退出Excel程序。
import xlwings as xw
app=xw.App(visible=True,add_book=False)
wb=app.books.add()
# wb就是新建的工作簿(workbook)，下面则对wb的sheet1的A1单元格赋值
wb.sheets['sheet1'].range('A1').value='人生'
wb.save()
wb.close()
app.quit()

#打开已保存的test.xlsx，在sheet2的第二个单元格输入“苦短”，然后保存关闭，退出Excel程序
import xlwings as xw
app=xw.App(visible=True,add_book=False)
wb=app.books.open(r'E:\\Github\\AdvancedPython\\xlwings\\test.xlsx')
# wb就是新建的工作簿(workbook)，下面则对wb的sheet1的A1单元格赋值
wb.sheets['sheet1'].range('A2').value='苦短'
wb.save()
wb.close()
app.quit()
#掌握以上代码，已经完全可以把Excel当作一个txt文本进行数据储存了，也可以读取Excel文件的数据，进行计算后，并将结果保存在Excel中。

### 引用工作簿、工作表和单元格
#1 引用工作簿，注意工作簿应该首先被打开(首先要将new_book_2.xlsx打开)
wb = xw.books['new_book_2.xlsx']
#2 引用活动工作簿
wb=xw.books.active
#3 引用工作簿中的sheet
sht=xw.books['new_book_2.xlsx'].sheets['first_sheet']
# 或者
wb=xw.books['new_book_2.xlsx']
sht=wb.sheets['first_sheet']
#4 引用活动sheet
sht = xw.sheets.active
#5 引用A1单元格
rng=xw.books['new_book_2.xlsx'].sheets['first_sheet'].range('A1')
#6 引用活动sheet上的单元格
# 注意Range首字母大写
rng=xw.Range('A1')

#3 其中需要注意的是单元格的完全引用路径是：
# 第一个Excel程序的第一个工作薄的第一张sheet的第一个单元格
xw.apps[0].books[0].sheets[0].range('A1')
#迅速引用单元格的方式是
sht = xw.books['new_book_2.xlsx'].sheets['first_sheet']
# A1单元格
rng=sht['A1']
# A1:B5单元格
rng=sht['A1:B5']
# 在第i+1行，第j+1列的单元格
# B1单元格
rng=sht[0,1]
# A1:J10
rng=sht[:10,:10]

##PS： 对于单元格也可以用表示行列的tuple进行引用
# A1单元格的引用
xw.Range((1,1))
#A1:C3单元格的引用
xw.Range((1,1),(3,3))

#3 储存数据
#储存单个值
# 注意".value“
sht.range('A1').value=1
#储存列表
# 将列表[1,2,3]储存在A1：C1中
sht.range('A1').value=[1,2,3]
# 将列表[1,2,3]储存在A1:A3中
sht.range('A1').options(transpose=True).value=[1,2,3] 
# 将2x2表格，即二维数组，储存在A1:B2中，如第一行1，2，第二行3，4
sht.range('A1').options(expand='table')=[[1,2],[3,4]] #报错:

## 读取数据
#读取单个值
# 将A1的值，读取到a变量中
a=sht.range('A1').value

#将值读取到列表中
#将A1到A2的值，读取到a列表中
a=sht.range('A1:A2').value
# 将第一行和第二行的数据按二维数组的方式读取
a=sht.range('A1:B2').value


"""
xlwings教程2:常用函数和方法
"""
###1 Book 工作簿常用的api
# 引用Excel程序中，当前的工作簿
wb = xw.books['new_book_2.xlsx']
# 返回工作簿的绝对路径
x=wb.fullname
# 返回工作簿的名称
x=wb.name
# 保存工作簿，默认路径为工作簿原路径，若未保存则为脚本所在的路径
x=wb.save(path=None)
# 关闭工作簿
x=wb.close()

"""
xlwings教程3
"""

"""
xlwings教程4

Python与VBA的比较:
非专业表哥，只是普通办公，希望偶尔遇到点重复的工作可以自动化一下。VBA对于我来说，要记得东西还是太多了，语法上不简洁。每写一个功能，代码太长了。
VBA虽然在很多程序都有接口，但是，应用范围还是略窄，对于一般用户深入学习后，但是，应用场景太少。有任务驱动，才有动力去提高水平。
Python运行速度绝对不算快的，但是，绝对比在Excel中运行VBA的速度还是要快很多
Python语言简洁（python大法好），模块丰富，有好多大神造好的轮子等着我们去用。
Python应用范围广，既能帮我解决偶尔遇到的表格问题，也能和其他各种软件或者平台联接起来。
"""
###运用Python自定义宏(仅限Windows)

    


