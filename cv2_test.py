# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:33:38 2019

@author: kyeoj
"""

import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

    
