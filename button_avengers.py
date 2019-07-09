import os,sys,time
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
from functools import partial
import cv2



# def click_button(button_flag, feature_list):
#     modeldir = './model/20180402-114759.pb'

#     def btn_event(idx, filename, name):
#         if (button_flag[idx] % 2 == 1):
#             button_flag[idx] += 1
#             if button_flag[idx] == 1:
#                 print ("Training Start")
#                 scale_img = prepro.collect_data(filename)
#                 obj = training(modeldir, scale_img, name)
#                 get_feature = obj.main_train()
#                 feature_list.append(get_feature)
#                 print('Getting feature map succeed')
#         elif(button_flag[idx] % 2 == 0):
#             button_flag[idx] += 1

width, height = 600, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.geometry("900x600+400+150")

main_frame = Frame(root)
main_frame.pack(fill = tk.X)

mv_label = tk.Label(main_frame)
mv_label.pack(side = tk.LEFT)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    mv_label.imgtk = imgtk
    mv_label.configure(image=imgtk)
    mv_label.after(10, show_frame)

show_frame()

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
