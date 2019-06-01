# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:27:24 2019

@author: kyeoj
"""

import augmentation_new
import os,sys,time
os.chdir(os.path.dirname(__file__))
from tkinter import *
import tkinter as tk
from classifier import training

#pre_image = []
#aug_img = augmentation_new.Make_Data('./avengers/robert', 'robert')
#pre_image.append(aug_img)

modeldir = './model/20170511-185253.pb'

        
def click_button(button_flag):

    def click1():    
        """
        respond to the button1 click
        """
        # toggle button colors as a test
        if (button_flag[1] % 2 ==1):
            button1.config(bg="white")
            button_flag[1] +=1 
            if button_flag[1] == 1:
                aug_img = augmentation_new.Make_Data('./avengers/evans', 'evans')
                #pre_image.append(aug_img)
                print ("Training Start")
                obj=training(modeldir, aug_img)
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
            
        elif (button_flag[1] %2 ==0):
            button1.config(bg="green")
            button_flag[1] +=1 
            
            
            
    def click2():
        """
        respond to the button2 click
        """
        # toggle button colors as a test
        if (button_flag[2] % 2 ==1):
            button2.config(bg="white")
            button_flag[2] +=1
            if button_flag[2] == 1:
                aug_img = augmentation_new.Make_Data('./avengers/hermsworth', 'hermsworth')
                #pre_image.append(aug_img)
                print ("Training Start")
                obj=training(modeldir, aug_img)
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
            
        elif(button_flag[2] %2 ==0):
            button2.config(bg="green")
            button_flag[2] += 1
            
    def click3():
        """
        respond to the button3 click
        """
        # toggle button colors as a test
        if (button_flag[3]%2==1):
            button3.config(bg="white")
            button_flag[3] += 1
            if button_flag[3] == 1:
                aug_img = augmentation_new.Make_Data('./avengers/jeremy', 'jeremy')
                #pre_image.append(aug_img)
                print ("Training Start")
                obj=training(modeldir, aug_img)
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
                
        
        elif(button_flag[3] %2 ==0):
            button3.config(bg="green")
            button_flag[3] += 1    
   
    def click4():
        """
        respond to the button4 click
        """    
        # toggle button colors as a test
        if (button_flag[4]%2 ==1):
            button4.config(bg="white")
            button_flag[4] += 1
            if button_flag[4] == 1:
                aug_img = augmentation_new.Make_Data('./avengers/mark', 'mark')
                #pre_image.append(aug_img)
                print ("Training Start")
                obj=training(modeldir, aug_img)
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
                
        
        elif(button_flag[4]%2 ==0):
            button4.config(bg="green")
            button_flag[4] += 1
            
            
    def click5():
        """
        respond to the button5 click
        """    
        # toggle button colors as a test
        if (button_flag[5] %2 ==1):
            button5.config(bg="white")
            button_flag[5] += 1
            if button_flag[5] == 1:
                aug_img = augmentation_new.Make_Data('./avengers/olsen', 'olsen')
                #pre_image.append(aug_img)
                print ("Training Start")
                obj=training(modeldir, aug_img)
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
                
            
        elif(button_flag[5] %2 ==0):
            button5.config(bg="green")
            button_flag[5] += 1
            
    
    root = tk.Tk()
    # create a frame and pack it
    frame1 = tk.Frame(root)
    frame1.pack(side=tk.TOP, fill=tk.X)
    
    # pick a (small) image file you have in the working directory ...
    filename1 = "C:/Facenet-Real-time-Tensorflow/avengers/evans/evans.jpg"
    photo1 = tk.PhotoImage(file=filename1)
    # create the image button, image is above (top) the optional text
    button1 = tk.Button(frame1, compound=tk.TOP, width=200, height=200, image=photo1,
                        text="optional text", bg='green', command=click1)
    button1.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button1.image = photo1

    filename2= "C:/Facenet-Real-time-Tensorflow/avengers/hermsworth/hermsworth.jpg"
    photo2 = tk.PhotoImage(file=filename2)
    # create the image button, image is above (top) the optional text
    button2 = tk.Button(frame1, compound=tk.TOP, width=200, height=200, image=photo2,
                        text="optional text", bg='green', command=click2)
    button2.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button2.image = photo2

    filename3= "C:/Facenet-Real-time-Tensorflow/avengers/jeremy/jeremy.jpg"
    photo3 = tk.PhotoImage(file=filename3)
    # create the image button, image is above (top) the optional text
    button3 = tk.Button(frame1, compound=tk.TOP, width=200, height=200, image=photo3,
                        text="optional text", bg='green', command=click3)
    button3.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button3.image = photo3
    
    filename4= "C:/Facenet-Real-time-Tensorflow/avengers/mark/mark.jpg"
    photo4 = tk.PhotoImage(file=filename4)
    # create the image button, image is above (top) the optional text
    button4 = tk.Button(frame1, compound=tk.TOP, width=200, height=200, image=photo4,
                        text="optional text", bg='green', command=click4)
    button4.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button4.image = photo4

    filename5= "C:/Facenet-Real-time-Tensorflow/avengers/olsen/olsen.jpg"
    photo5 = tk.PhotoImage(file=filename5)
    # create the image button, image is above (top) the optional text
    button5 = tk.Button(frame1, compound=tk.TOP, width=200, height=200, image=photo5,
                        text="optional text", bg='green', command=click5)
    button5.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button5.image = photo5
    # start the event loop
    root.mainloop()