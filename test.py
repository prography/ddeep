# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:44:15 2019

@author: kyeoj
"""
import os
path = os.path.expanduser('~Facenet-Real-time-Tensorflow/avengers/evans/evans.gif')
ab_path = os.path.dirname(os.path.abspath(__file__))
print(path)
print(ab_path)
print(os.path.join(ab_path,'avengers/evans/evans.gif'))
print(os.getcwd())