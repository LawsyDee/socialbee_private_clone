# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:00:43 2016

@author: Laura Drummer
"""

import os
from collections import Counter

root_dir = 'F:\\Data\\enron_mail_20150507\\maildir\\'
all_words=[]
counted_words = {}
for folder, subs, files in os.walk(root_dir):
    for filename in files:
        with open(os.path.join(folder,filename),'r') as src:
            data = src.read()
            data = data.replace('\t', ' ')
            data = data.replace('\n', ' ')
            data = data.replace('!', ' ')
            data = data.replace("'", '')
            data = data.replace('.', ' ')
            data = data.replace(',', ' ')
            data = data.replace(':', ' ')
            data = data.replace(';', ' ')
            data = data.replace('"', ' ')
            data = data.replace('#', ' ')
            data = data.replace('(', '')
            data = data.replace(')', '')
            token_words = [word.lower() for word in data.split(' ') if (word != ' ' and word != '')]
        for w in token_words:
            if w not in counted_words.keys():
                counted_words[w]=0
            counted_words[w]+=1

total_count = Counter(counted_words)

with open("final_count.txt", "w") as fcf:
    for k,v in total_count.most_common(500):
        fcf.write(k+"\t"+str(v)+'\n')


#>>> len(total_count)
#2163220
#>>> counted_words['enron']
#612654
#>>> counted_words['laura']
#32237
#>>> counted_words['drummer']
#27
#>>> total_count.most_common(50)
#[('com', 5728589), ('the', 5604336), ('to', 4477409), ('and', 2536028), 
# ('of', 2353818), ('a', 1992516), ('in', 1717162), ('for', 1470978), 
#('>', 1355304), ('from', 1251423), ('on', 1234907), ('is', 1214044), 
#('you', 1134315), ('that', 1062701), ('i', 981868), ('subject', 959402), 
#('this', 891723), ('be', 836231), ('com>', 792561), ('1', 783321), 
#('with', 762916), ('will', 736746), ('00', 707175), ('at', 698444), 
#('-', 693896), ('have', 692015), ('by', 673741), ('we', 666634), 
#('are', 647669), ('0', 628208), ('it', 621083), ('as', 618224), 
#('date', 613302), ('enron', 612654), ('=20', 592351), ('or', 577764),
# ('2001', 542708), ('content-type', 522787), ('mime-version', 521959), 
#('message-id', 521359), ('text/plain', 519553), 
#('content-transfer-encoding', 518949), ('javamail', 517437), ('x-to', 517406), 
#('x-origin', 517401), ('evans@thyme>', 517401), ('x-cc', 517401), 
#('x-bcc', 517401), ('x-from', 517401), ('x-filename', 517401)]