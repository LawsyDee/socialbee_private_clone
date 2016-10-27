# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:30:25 2016

@author: Laura Drummer
"""

import os

root_dir = 'F:\\Data\\enron_mail_20150507\\maildir'

enron_data_dict={}

users = os.listdir(root_dir)
print("parsing emails")

for user in users:
    print(user)
    enron_data_dict[user] = {'foldernames':[],'total_emails':0}
    for folder, subs, files in os.walk(root_dir+'\\'+user):
        fn = folder.split('\\')[-1]
        if fn not in enron_data_dict[user]['foldernames']:
            enron_data_dict[user]['foldernames'].append(fn)
            enron_data_dict[user]['total_emails']+=len(files)
