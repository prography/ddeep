from flask import Flask,render_template,jsonify, request
import os,sys,time
#os.chdir(os.path.dirname(__file__))
from PIL import Image
from PIL import ImageTk
from classifier import training
import preprocess as prepro
import json
import facenet
import tensorflow as tf
import numpy as np

app = Flask(__name__)
modeldir = './model/20180402-114759.pb'
feature_list = []

sess = tf.Session()
with sess.as_default():
    facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
feature_arr = 0
feat = False

# 학습 버튼을 눌렀을 때 한명을 학습하고 그 feature를 feature_arr로 저장
@app.route('/learn',methods=["POST"])
def button_train():
    global feature_arr,feat
    feature_arr = 0
    print("=========================================================================================")
    img = request.json['face_list']
    img_np =np.array(img)

    obj = training(modeldir, img_np)
    feature_arr = obj.main_train()
    feat = True

    return "Success!"

@app.route('/video', methods = ["POST"])
def video_feature():
    val = request.json
    value = val['images_placeholder']
    np.set_printoptions(suppress=True,precision=20)
    scaled_reshape = np.array(value)
    global feat,feature_arr
    
    embedding_arr = np.zeros((1, embedding_size))
    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    embedding_arr[0, :] = sess.run(embeddings, feed_dict=feed_dict)
    #사람이 등록되었을 경우와 등록 안되어있는 경우를 조건을 다르게 처리함
    if feat:
        img_data = facenet.check_features(feature_arr[0], embedding_arr[0])
    else:
        img_data = {"name": "", "cos_sim" : 0}
        
    print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])
    
    return jsonify(img_data)
    

if __name__ =='__main__':
   #app.run(host='0.0.0.0',port=5000,debug=True)
   app.run(debug=True)
   #서버 실행하는 부분. 
