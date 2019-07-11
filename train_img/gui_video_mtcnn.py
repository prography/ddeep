##################################
#                                #
#     identify랑 gui부분 연결함.  #                                  
#                                #   
##################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import pickle
import threading
import os,sys,time

from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
from functools import partial
import mtcnn
import json
import requests

input_video="captain.mp4"
modeldir = './model/20180402-114759.pb'
npy='./npy'

button_flag  = [1,1,1,1,1,1]
feature_list = []
button_name = ['','evans','hermsworth','jeremy','mark','olsen']

def btn_event(idx):
    if (button_flag[idx] % 2 == 1):
        button[idx].config(bg = "white")
        
        if button_flag[idx] == 1:
            print("training start")
            URL = "http://127.0.0.1:5000/button/"+name[idx]
            print(URL)
            response = requests.get(URL)
            print('Getting feature map succeed')
            
    elif(button_flag[idx] % 2 == 0):
        button[idx].config(bg="green")
    
    button_flag[idx] += 1

#GUI
width, height = 600, 600


root = tk.Tk()
root.geometry("900x600+400+150")

main_frame = tk.Frame(root)
main_frame.pack(fill = tk.X)

mv_label = tk.Label(main_frame)
mv_label.pack(side = tk.LEFT)

face_label = tk.Label(main_frame)
face_label.pack(side = tk.LEFT)


detector = mtcnn.MTCNN()
image_size = detector.image_size
input_image_size = detector.input_image_size

sess = tf.Session()
print('Loading Modal')
with sess.as_default():
    facenet.load_model(modeldir)

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

cap = cv2.VideoCapture(input_video)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print('Start Recognition')
def show_frame():
    print(button_flag)
    _, cv2image = cap.read()
    
    bounding_boxes, cv2image = detector.run_mtcnn(cv2image)
    nrof_faces = bounding_boxes.shape[0]
    print('Detected_FaceNum: %d' % nrof_faces)


    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(cv2image.shape)[0:2]

        cropped = []
        scaled = [] 
        scaled_reshape = []
        bb = np.zeros((nrof_faces,4), dtype = np.int32)

        for i in range(nrof_faces):
            emb_array = np.zeros((1, embedding_size))

            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

    #여기부터 add ------------>  
            face = cv2image[bb[i][1] : bb[i][3], 
                            bb[i][0] : bb[i][2]]
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(cv2image[0]) or bb[i][3] >= len(cv2image):
                print('Face is very close! 0:',bb[i][0],'    1:',bb[i][1],'      2:',bb[i][2],'          3:',bb[i][3])
                continue
                
            cropped.append(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cropped[i] = facenet.flip(cropped[i], False)

            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                   interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
            
        #서버로 넘김. 
            #URL = "http://127.0.0.1:5000/video"
            #tolist_img = scaled_reshape[i].tolist()
            #json_feed = {'images_placeholder': tolist_img}
            #response = requests.post(URL, data = json_feed)
        #확인 
            #print("scaled_reshape type: ",type(scaled_reshape[i]))
            #print("scaled_reshape shape", scaled_reshape[i].shape)
            #print(scaled_reshape[i])
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
            #print("emb_array type:", type(emb_array))
            
            img_data = facenet.check_features(feature_list, emb_array[0], {"name" : "", "cos_sim" : 0}, 0)

            
            print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])
    ##########################################################################################################  
    #                                                                                                        # 
    #  현재 GUI에서 button부분이랑 연결이 안되서 우선 이렇게 밖으로 뺴서 얼굴부분은 모두 모자이크 처리하도록 했으요  # 
    #                                                                                                        #               
    ##########################################################################################################                                                                                                       
            cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))
           
            if img_data["cos_sim"] >= 0.5:
                
                if button_flag[button_name.index(img_data["name"])]%2 == 0:
                    cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                    
                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    cv2.putText(cv2image, img_data["name"], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))
                
            else:                           
                cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]  

    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    #cv2image = cv2.flip(cv2image, 1)
    
    #face = cv2.flip(face, 1)
    #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
    webcam_img = ImageTk.PhotoImage(image = Image.fromarray(cv2image))
    mv_label.imgtk = webcam_img
    mv_label.configure(image = webcam_img)
    mv_label.after(10, show_frame)

    #face_img = ImageTk.PhotoImage(image = Image.fromarray(face))
    #face_label.imgtk = face_img
    #face_label.configure(image = face_img)

show_frame()



#button 부분. 
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
    button.append(tk.Button(photo_frame, width = 130, height = 130, image = photo[idx], text = name[idx],
                            command = partial(btn_event, idx)))
    button[idx].pack(side = tk.LEFT, padx = 25, pady = 10)
    button[idx].image = photo[idx]

root.mainloop()