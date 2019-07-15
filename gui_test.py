from tkinter import *
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image
from PIL import ImageTk
import cv2
import mtcnn
import numpy as np

width, height = 800, 800
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = mtcnn.MTCNN()

blur_check = True # 블러 처리를 할건지 얼굴을 찾아내고 학습할건지 여부 결정, show_frame 함수 아래쪽에서 사용

def blur_face():
    global blur_check
    blur_check = True # 현재 blur 상태라는 것을 알림.
    blur_btn.configure(state = "disabled") # blur처리를 누르고 난 뒤 blur 버튼 비활성화
    learn_btn.configure(state = "disabled")
    detect_btn.configure(state = "active") # 얼굴을 찾는 버튼 활성화

def detect_face():
    global blur_check
    blur_check = False # 현재 얼굴을 찾아 rectangle로 표시하는 과정을 진행중임.
    detect_btn.configure(state = "disabled") # detect_btn 비활성화
    blur_btn.configure(state = "active") # blur 처리로 다시 돌아갈 수 있도록 활성화.
    learn_btn.configure(state = "active") # 학습 버튼 활성화.

def learn_face(): # 학습 버튼 이벤트 함수. 서버와 연결하면 될듯.
    global blur_check
    blur_check = True # 학습 버튼을 누르게 되면 자동적으로 웹캠이 blur 처리 되고 확인할 수 있도록 변경.
    learn_btn.configure(state = "disabled") # 학습 버튼 비활성화.
    

# GUI 위젯 및 레이아웃
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
detect_btn = tk.Button(root, text = "Face Detect", width = 10, height = 8, command = detect_face) 
detect_btn.place(x = 350, y = 420)

# 얼굴을 찾고 난뒤 이 버튼을 누르면 잡아낸 얼굴을 서버로 보내도록 하면됨.
learn_btn = tk.Button(root, text = "Learn", state = "disabled", bg = "green", width = 10, height = 8, command = learn_face)
learn_btn.place(x = 460, y = 420)

# 블러 처리하는 버튼. 처음 시작할 때 블러 처리하며, 학습 하고 난 뒤 내 얼굴을 잘 학습했는지 확인할때도 사용함.
blur_btn = tk.Button(root, text = "Blur", state = "disabled", width = 10, height = 8, command = blur_face)
blur_btn.place(x = 580, y = 420)
    
# 웹캠과 얼굴 처리 구간
def show_frame():
    _, cv2image = cap.read()

    global blur_check
    
    bounding_boxes, cv2image = detector.run_mtcnn(cv2image)
    nrof_faces = bounding_boxes.shape[0]

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((nrof_faces, 4), dtype = np.int32)

        for i in range(nrof_faces):
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            if blur_check: # True라면 웹캠에 나오는 모든 얼굴 blur 처리
                cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]] = cv2.blur(cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]], (23,23))
            else: # False라면 웹캠에 나오는 모든 얼굴 위치 좌표에 초록색 사각형 그리기
                cv2.rectangle(cv2image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 3)
            
            face = cv2image[bb[i][1] : bb[i][3], bb[i][0] : bb[i][2]]

    cv2image = cv2.flip(cv2image, 1)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    
    # 웹캠 프레임을 mv_label에 띄우는 부분
    webcam_img = ImageTk.PhotoImage(image = Image.fromarray(cv2image))
    mv_label.imgtk = webcam_img
    mv_label.configure(image = webcam_img)
    mv_label.after(10, show_frame)

    # 왼쪽 아래 얼굴 이미지를 crop해서 띄울건데, blur 처리 된 경우에는 띄우지 않고 rectangle 처리 됐을때만 띄우도록.
    if nrof_faces > 0 and not blur_check:
        if nrof_faces > 1: # 만약 rectangle 처리된 얼굴이 2개 이상이라면 label로 경고 글씨 표시
            warning_label.configure(text = "한 개의 얼굴만 나오도록 해주세요.")
            learn_btn.configure(state = "disabled")
        else: # 그렇지 않다면 지우고.
            warning_label.configure(text = "")
            learn_btn.configure(state = "active")
        face = cv2.flip(face, 1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
        face_img = ImageTk.PhotoImage(image = Image.fromarray(face))
        face_label.imgtk = face_img
        face_label.configure(image = face_img)

show_frame()

root.mainloop()