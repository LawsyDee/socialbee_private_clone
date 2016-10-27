# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:20:13 2016

@author: Laura Drummer
"""
er_file = "rough_entity_resolution.txt"

with open(er_file) as er:
    data = er.readlines()
    data = [row.rstrip().split('\t') for row in data]

data_dict = {}

for row in data:
    if row[1] not in data_dict.keys():
        data_dict[row[1]] = [row[0]]
    else:
        data_dict[row[1]].append(row[0])

for name in data_dict.keys():
    if len(data_dict[name]) > 1:
        print(name, data_dict[name])