import csv
import sys
fn=sys.argv[1]+'.csv'
rows=[]
with open(fn,newline='') as csvf:
    rows=list(csv.reader(csvf))
otf=open('t'+fn,'w')
for i in range(len(rows)):
    print('0.0,0.0,',rows[i][1],sep='',file=otf)
fn=sys.argv[1]+'_train.csv'
rows=[]
with open(fn,newline='') as csvf:
    rows=list(csv.reader(csvf))
otf=open('t'+fn,'w')
for i in range(len(rows)):
    print('0.0,0.0,',rows[i][1],sep='',file=otf)
fn=sys.argv[1]+'_val.csv'
rows=[]
with open(fn,newline='') as csvf:
    rows=list(csv.reader(csvf))
otf=open('t'+fn,'w')
for i in range(len(rows)):
    print('0.0,0.0,',rows[i][1],sep='',file=otf)
