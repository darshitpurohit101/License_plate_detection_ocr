#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:58:57 2021

@author: darshit
"""

import pytesseract as ps
import cv2

def to_text(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (300,200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.Canny(img,80,350)
    #img = cv2.threshold(img, 0, 500, cv2.THRESH_BINARY )[1]
    
    custom_config = r'-l grc+tha+eng --psm 6'
    print("text: ",ps.image_to_string(img, config=custom_config))

#    cv2.imshow('test',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()