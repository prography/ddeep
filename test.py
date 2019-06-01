# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:44:15 2019

@author: kyeoj
"""
import os
path = os.path.expanduser('./avengers/evans')
img_path = os.path.join(path, path.split("/")[-1] + ".jpg")
print(img_path)