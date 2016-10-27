# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:42:13 2016

@author: Laura Drummer
"""
import config 
import os

root_dir = config.data_path

message_dict={}
comms_list = []
art_network={}
print("parsing emails")
for folder, subs, files in os.walk(root_dir):
    for filename in files:
      with open(folder+'\\'+filename) as f:
          data = f.read()
          if "comments on el paso proposal to allocate receipt point rights" in data.lower():
              print(folder, filename)