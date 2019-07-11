import os,sys,time
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
from functools import partial
import cv2
import mtcnn
import numpy as np



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

detector = mtcnn.MTCNN()

root = tk.Tk()
root.geometry("900x600+400+150")

main_frame = tk.Frame(root)
main_frame.pack(fill = tk.X)

mv_label = tk.Label(main_frame)
mv_label.pack(side = tk.LEFT)

face_label = tk.Label(main_frame)
face_label.pack(side = tk.LEFT)

def show_frame():
    _, cv2image = cap.read()
    
    bounding_boxes, cv2image = detector.run_mtcnn(cv2image)
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((nrof_faces,4), dtype = np.int32)

        for i in range(nrof_faces):
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 3) 
    
            face = cv2image[bb[i][1] : bb[i][3], 
                            bb[i][0] : bb[i][2]]
    
    cv2image = cv2.flip(cv2image, 1)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    face = cv2.flip(face, 1)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
    webcam_img = ImageTk.PhotoImage(image = Image.fromarray(cv2image))
    mv_label.imgtk = webcam_img
    mv_label.configure(image = webcam_img)
    mv_label.after(10, show_frame)

    face_img = ImageTk.PhotoImage(image = Image.fromarray(face))
    face_label.imgtk = face_img
    face_label.configure(image = face_img)

show_frame()

sub_filename = ['avengers/evans/evans.gif', 'avengers/hermsworth/hermsworth.gif', 'avengers/jeremy/jeremy.gif',
                'avengers/mark/mark.gif', 'avengers/olsen/olsen.gif']
name = ["evans", "hermsworth", "jeremy", "mark", "olsen"]
filename = []
photo = []
button = []

photo_frame = Frame(root)
photo_frame.pack(fill = tk.X)

for idx in range(len(sub_filename)):
    filename.append(os.path.join(os.getcwd(), sub_filename[idx]))
    photo.append(tk.PhotoImage(file = filename[idx]).subsample(5))
    button.append(tk.Button(photo_frame, width = 130, height = 130, image = photo[idx], text = name[idx]))
    button[idx].pack(side = tk.LEFT, padx = 25, pady = 10)
    button[idx].image = photo[idx]

root.mainloop()