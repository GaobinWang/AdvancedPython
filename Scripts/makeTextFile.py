#!/usr/bin/env python3

"makeTextFile.py---this module is to create a file"

import os 
ls=os.linesep

#while True:

fname=input("please input the file name:")
#print(fname)

if os.path.exists(fname):
   print("Error: %s aready exists" % fname)
else:
   print("sucess")

#get file content lines
all=[]
print("\nEnter lines ('.'  byitself to quit  ).\n")

#loop until terminates input
while True:
   entry=input(">")
   if entry=='.':
      break
   else:
      all.append(entry)
print(all)

#write lines to file with proper line-ending
fobj=open(fname,'w')
fobj.writelines(["%s%s" % (x,ls) for x in all])
fobj.close()
print("Done!")
