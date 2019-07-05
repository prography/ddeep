import os,sys,time
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
from functools import partial


def click_button(button_flag, feature_list):
    modeldir = './model/20180402-114759.pb'

    def btn_event(idx, filename):
        if (button_flag[idx] % 2 == 1):
            button_flag[idx] += 1
            if button_flag[idx] == 1:
                print ("Training Start")
                scale_img = prepro.collect_data(filename)
                obj = training(modeldir, scale_img, "olsen")
                get_feature = obj.main_train()
                feature_list.append(get_feature)
                print('Getting feature map succeed')
        elif(button_flag[idx] % 2 == 0):
            button_flag[idx] += 1

    root = tk.Tk()
    root.geometry("1200x800+300+50")

    main_frame = Frame(root)
    main_frame.pack(fill = tk.X)

    mv_label = tk.Label(main_frame, text = "MOVIE", width = 105, height = 40, fg="red", relief="solid")
    mv_label.pack(side = tk.LEFT)

    sub_filename = ['avengers/evans/evans.gif', 'avengers/hermsworth/hermsworth.gif', 'avengers/jeremy/jeremy.gif',
                    'avengers/mark/mark.gif', 'avengers/olsen/olsen.gif']
    name = ["evans", "hermsworth", "jeremy", "mark", "olsen"]
    filename = []
    photo = []
    button = []

    arrow_img = tk.PhotoImage(file = os.path.join(os.getcwd(), "arrow.png")).subsample(6)
    start_btn = tk.Button(main_frame, width = 130, height = 130, image = arrow_img)
    start_btn.pack(side = tk.LEFT, padx = 80)
    start_btn.image = arrow_img

    photo_frame = Frame(root)
    photo_frame.pack(fill = tk.X)

    for idx in range(len(sub_filename)):
        filename.append(os.path.join(os.getcwd(), sub_filename[idx]))
        photo.append(tk.PhotoImage(file = filename[idx]).subsample(5))
        button.append(tk.Button(photo_frame, width = 130, height = 130, image = photo[idx], text = name[idx]))
        button[idx].pack(side = tk.LEFT, padx = 25, pady = 10)
        button[idx].image = photo[idx]

    root.mainloop()