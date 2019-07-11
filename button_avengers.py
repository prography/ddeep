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
import requests

button_flag = [1,1,1,1,1,1]

#btn_event수정 flask랑 연결했음. 버튼만 잘 동작하게 만들면 서버는 잘 돌아갈듯
def btn_event(idx):
    if (button_flag[idx] % 2 == 1):
        button[idx].config(bg="white")
        
        if button_flag[idx] == 1:
            print("training start")
            URL = "http://127.0.0.1:5000/button/"+name[idx]
            print(URL)
            response = requests.get(URL)
            print('Getting feature map succeed')
            
    elif(button_flag[idx] % 2 == 0):
        button[idx].config(bg="green")
    
    button_flag[idx] += 1

input_video="captain.mp4"
width, height = 600, 600
cap = cv2.VideoCapture(input_video)
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
    face = False
    
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
    
#    cv2image = cv2.flip(cv2image, 1)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    
    webcam_img = ImageTk.PhotoImage(image = Image.fromarray(cv2image))
    mv_label.imgtk = webcam_img
    mv_label.configure(image = webcam_img)
    mv_label.after(10, show_frame)
    
    if face:
        face = cv2.flip(face, 1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
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
    #bg = 'green', command=lambda: btn_event(idx) 추가함
    button.append(tk.Button(photo_frame, width = 130, height = 130, image = photo[idx], text = name[idx], bg = 'green', command=lambda: btn_event(idx)))
    button[idx].pack(side = tk.LEFT, padx = 25, pady = 10)
    button[idx].image = photo[idx]

root.mainloop()