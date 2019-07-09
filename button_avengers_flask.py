import os,sys,time
#os.chdir(os.path.dirname(__file__))
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
import requests
import json
import numpy as np


        
def click_button(button_flag, feature_list):
    def click1():
        """
        respond to the button1 click
        """
        # toggle button colors as a test
        if (button_flag[1] % 2 ==1):
            button1.config(bg="white")
            
            if button_flag[1] == 1:
                URL = "http://127.0.0.1:5000/button/evans"
                response = requests.get(URL)
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
                #pre_image.append(aug_img)
                print ("Training Start")
                scale_img = prepro.collect_data(os.path.join(os.getcwd(),'avengers/hermsworth/hermsworth_p.jpg'))
                obj = training(modeldir, scale_img, "hermsworth")
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
                #pre_image.append(aug_img)
                print ("Training Start")
                scale_img = prepro.collect_data(os.path.join(os.getcwd(),'avengers/jeremy/jeremy_p.jpg'))
                obj = training(modeldir, scale_img, "jeremy")
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
                #pre_image.append(aug_img)
                print ("Training Start")
                scale_img = prepro.collect_data(os.path.join(os.getcwd(),'avengers/mark/mark_p.jpg'))
                obj = training(modeldir, scale_img, "mark")
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
                #pre_image.append(aug_img)
                print ("Training Start")
                scale_img = prepro.collect_data(os.path.join(os.getcwd(),'avengers/olsen/olsen_p.jpg'))
                obj = training(modeldir, scale_img, "olsen")
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
    filename1 = os.path.join(os.getcwd(),'avengers/evans/evans.gif')
    photo1 = tk.PhotoImage(file=filename1).subsample(3)#--------
    # create the image button, image is above (top) the optional text
    button1 = tk.Button(frame1, compound=tk.TOP, width=150, height=150, image=photo1,
                        text="evans", bg='green', command=click1)
    button1.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button1.image = photo1
    
    filename2 = os.path.join(os.getcwd(), 'avengers/hermsworth/hermsworth.gif')
    '''
    pil_photo2 = Image.open(filename2)
    pil_photo2 = pil_photo2.resize((200,200), Image.ANTIALIAS)
    photo2 = tk.PhotoImage(pil_photo2)
    '''
    photo2 = tk.PhotoImage(file=filename2).subsample(3)
    
    # create the image button, image is above (top) the optional text
    button2 = tk.Button(frame1, compound=tk.TOP, width=150, height=150, image=photo2,
                        text="hermsworth", bg='green', command=click2)
    button2.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button2.image = photo2
    
    filename3= os.path.join(os.getcwd(), 'avengers/jeremy/jeremy.gif')
    photo3 = tk.PhotoImage(file=filename3).subsample(3)
    # create the image button, image is above (top) the optional text
    button3 = tk.Button(frame1, compound=tk.TOP, width=150, height=150, image=photo3,
                        text="jeremy", bg='green', command=click3)
    button3.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button3.image = photo3
    
    filename4= os.path.join(os.getcwd(), 'avengers/mark/mark.gif')
    photo4 = tk.PhotoImage(file=filename4).subsample(3)
    # create the image button, image is above (top) the optional text
    button4 = tk.Button(frame1, compound=tk.TOP, width=150, height=150, image=photo4,
                        text="mark", bg='green', command=click4)
    button4.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button4.image = photo4
    
    filename5 = os.path.join(os.getcwd(), 'avengers/olsen/olsen.gif')
    photo5 = tk.PhotoImage(file=filename5).subsample(6)
    # create the image button, image is above (top) the optional text
    button5 = tk.Button(frame1, compound=tk.TOP, width=150, height=150, image=photo5,
                        text="olsen", bg='green', command=click5)
    button5.pack(side=tk.LEFT, padx=4, pady=50)
    # save the button's image from garbage collection (needed?)
    button5.image = photo5
    # start the event loop
    root.mainloop()
