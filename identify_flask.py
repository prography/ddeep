from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from button_avengers_flask import click_button
import threading
import mtcnn
from flask import jsonify
import json
import requests

button_flag  = [1,1,1,1,1,1]
feature_list = []
button_name = ['','evans','hermsworth','jeremy','mark','olsen']
th = threading.Thread(target = click_button, args = (button_flag, feature_list))
th.daemon = True
th.start()

input_video="captain.mp4"
modeldir = './model/20180402-114759.pb'
npy='./npy'

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

video_capture = cv2.VideoCapture(input_video)


print('Start Recognition')
while True:
    print(button_flag)
    ret, frame = video_capture.read()
    
    bounding_boxes, frame = detector.run_mtcnn(frame)
    
    nrof_faces = bounding_boxes.shape[0]
    print('Detected_FaceNum: %d' % nrof_faces)

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(frame.shape)[0:2]

        cropped = []
        scaled = []
        scaled_reshape = []
        bb = np.zeros((nrof_faces,4), dtype=np.int32)

        for i in range(nrof_faces):
            #emb_array = np.zeros((1, embedding_size))

            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]
            
            

            # inner exception
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                print('Face is very close! 0:',bb[i][0],'    1:',bb[i][1],'      2:',bb[i][2],'          3:',bb[i][3])
                continue
                
            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cropped[i] = facenet.flip(cropped[i], False)
            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                   interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
            
            URL = "http://127.0.0.1:5000/video"
            tolist_img = scaled_reshape[i].tolist()
            tolist_feature = [x.tolist]
            json_feed = {'images_placeholder': tolist_img}
            response = requests.post(URL, data = json_feed)
            
            '''
            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
            
            img_data = facenet.check_features(feature_list, emb_array[0], {"name" : "", "cos_sim" : 0}, 0)
            '''
            
            img_data = json.loads(response.json())
            
            print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])

            if img_data["cos_sim"] >= 0.5:
              
                if button_flag[button_name.index(img_data["name"])]%2 == 0:
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                    
                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    cv2.putText(frame, img_data["name"], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))
                
            else:                           
                frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
