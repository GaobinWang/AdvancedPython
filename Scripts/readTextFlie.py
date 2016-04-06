#!/usr/bin/env python3

"readTextFile.py---read test file"

fname=input("please input the file name which you want to read and print:")
print(fname)

try:
   fobj=open(fname,'r')
except:
   print("### open file fail")
else:
   for i in fobj:
      print(i)
   fobj.close()

