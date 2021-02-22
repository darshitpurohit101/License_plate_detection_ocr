# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:14:47 2021

@author: 91820
"""

import urllib.request

file = "number_plate.txt"

f = open(file)
lines = f.readlines()
urls = []

for line in lines:
    url=""
    for chr in line[13:]:
        if chr != '"':
            url = url + chr
        if chr == '"':
            break
    urls.append(url)

file_no = 0
for path in urls:
    r = urllib.request.urlopen(path)
    with open("G:\\\cars\\car"+str(file_no)+".jpg", "wb") as f:
        f.write(r.read())
    file_no+=1