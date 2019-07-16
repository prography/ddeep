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
import tkinter.font as tkfont
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

server = "http://127.0.0.1:5000/"
modeldir = './model/20180402-114759.pb'
npy='./npy'
feature_list = []

########## GUI

width, height = 800, 800
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

blur_check = True
face = 0
learn_check = False
def blur_face_btn():
    global blur_check
    blur_check = True                      # 현재 blur 상태라는 것을 알림.
    blur_btn.configure(state = "disabled") # blur처리를 누르고 난 뒤 blur 버튼 비활성화
    learn_btn.configure(state = "disabled")
    detect_btn.configure(state = "active") # 얼굴을 찾는 버튼 활성화

def detect_face_btn():
    global blur_check
    blur_check = False                      # 현재 얼굴을 찾아 rectangle로 표시하는 과정을 진행중임.
    detect_btn.configure(state = "disabled")# detect_btn 비활성화
    blur_btn.configure(state = "active")    # blur 처리로 다시 돌아갈 수 있도록 활성화.
    learn_btn.configure(state = "active")   # 학습 버튼 활성화.


def learn_face_btn():                       # 학습 버튼 이벤트 함수. 서버와 연결하면 될듯.
    global blur_check
    global face                             # 학습 버튼을 누르게 되면 자동적으로 웹캠이 blur 처리 되고 확인할 수 있도록 변경.
   
    global learn_check
    learn_check=True
    blur_check = True
    learn_btn.configure(state = "disabled") # 학습 버튼 비활성화.



root = tk.Tk()
root.geometry("900x650+400+150")

# 웹캠 화면.
mv_label = tk.Label(root)
mv_label.pack(fill = tk.X, side = tk.TOP)

# 얼굴이 2개이상일 경우 나타나는 경고 레이블.
warning_font = tkfont.Font(family = "궁서체", size = 20)
warning_label = tk.Label(root, font = warning_font, fg = "red")
warning_label.place(x = 280, y = 370)

# Crop한 얼굴 이미지를 띄우기 위한 레이블. detect_btn을 누르고 얼굴을 찾아냈을 때만 나타남.
face_label = tk.Label(root)
face_label.place(x = 100, y = 400)

# 얼굴을 인식하도록 rectangle 처리 하는 버튼. 이 버튼을 누른 후에 얼굴을 학습하는 learn_btn이 활성화됨.
detect_btn = tk.Button(root, text = "Face Detect", width = 10, height = 8, command = detect_face_btn) 
detect_btn.place(x = 350, y = 420)

# 얼굴을 찾고 난뒤 이 버튼을 누르면 잡아낸 얼굴을 서버로 보내도록 하면됨.
learn_btn = tk.Button(root, text = "Learn", state = "disabled", bg = "green", width = 10, height = 8, command = learn_face_btn)
learn_btn.place(x = 460, y = 420)

# 블러 처리하는 버튼. 처음 시작할 때 블러 처리하며, 학습 하고 난 뒤 내 얼굴을 잘 학습했는지 확인할때도 사용함.
blur_btn = tk.Button(root, text = "Blur", state = "disabled", width = 10, height = 8, command = blur_face_btn)
blur_btn.place(x = 580, y = 420)



########### mtcnn
detector = mtcnn.MTCNN()
image_size = detector.image_size
input_image_size = detector.input_image_size

sess = tf.Session()
print('Loading Modal')
#triplet loss 학습 모델 facenet에서 load model.
with sess.as_default():
    facenet.load_model(modeldir)

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


print('Start Recognition')

def show_frame():
    _, cv2image = cap.read()
    global blur_check,learn_check
    #inception_resnet 실행.
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
            

            if blur_check: # True라면 웹캠에 나오는 모든 얼굴 blur 처리
                cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]] = cv2.blur(cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]], (23,23))
            else: # False라면 웹캠에 나오는 모든 얼굴 위치 좌표에 초록색 사각형 그리기
                cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 3)
                global face
                face = cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]]
            
    #여기부터 add ------------>  
            
            # if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(cv2image[0]) or bb[i][3] >= len(cv2image):
            #     print('Face is very close! 0:',bb[i][0],'    1:',bb[i][1],'      2:',bb[i][2],'          3:',bb[i][3])
            #     continue
                
            cropped.append(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cropped[i] = facenet.flip(cropped[i], False)

            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                   interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
            
            
        #서버로 넘김. 
            URL = server+"video"
            tolist_img = scaled_reshape[i].tolist()
            json_feed = {'images_placeholder': tolist_img}
            response = requests.post(URL, json = json_feed)
            
            img_data = response.json()
            if learn_check:
                URL= server+"learn"
                print('Getting feature map succeed')        
                json_feed = {'face_list':tolist_img}
                response=requests.post(URL,json=json_feed)
                learn_check=False
        #확인 

            print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])

        
            if img_data["cos_sim"] >= 0.5 and blur_check:
                cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                    
                    #plot result idx under box
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                cv2.putText(cv2image, "Me!", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255, 255, 255), thickness=1, lineType=2)
                """
                if button_flag[button_name.index(img_data["name"])]%2 == 0:
                    cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                    
                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    cv2.putText(cv2image, img_data["name"], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (255, 255, 255), thickness=1, lineType=2)
                else:
                    cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))
                """
            # else:  이부분이 문제였음. 아마 코사인 유사도가 0이기 때문에 이부분도 같이 실행되서 그런듯.
            #     cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(cv2image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))

    cv2image = cv2.flip(cv2image, 1)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    
   
    webcam_img = ImageTk.PhotoImage(image = Image.fromarray(cv2image))
    mv_label.imgtk = webcam_img
    mv_label.configure(image = webcam_img)
    mv_label.after(10, show_frame)

    # 왼쪽 아래 얼굴 이미지를 crop해서 띄울건데, blur 처리 된 경우에는 띄우지 않고 rectangle 처리 됐을때만 띄우도록.
    if nrof_faces > 0 and not blur_check:
        if nrof_faces > 1: # 만약 rectangle 처리된 얼굴이 2개 이상이라면 label로 경고 글씨 표시
            warning_label.configure(text = "등록을 원하시면, 한 명의 얼굴만 나오도록 해주세요.")
            learn_btn.configure(state = "disabled")
        else: # 그렇지 않다면 경고문을 지우고.
            warning_label.configure(text = "")
            learn_btn.configure(state = "active")
        face = cv2.flip(face, 1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
        face_img = ImageTk.PhotoImage(image = Image.fromarray(face))
        face_label.imgtk = face_img
        face_label.configure(image = face_img)
show_frame()

root.mainloop()
