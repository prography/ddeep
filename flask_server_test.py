#flask_server ver2 
from flask import Flask,render_template,jsonify, request
import os,sys,time
#os.chdir(os.path.dirname(__file__))
from tkinter import *
import tkinter as tk
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

@app.route('/button/<name>')
def button_train(name):
    scale_img = prepro.collect_data(os.path.join(os.getcwd(),'avengers/'+name+'/'+name+'_p.jpg'))
    obj = training(modeldir, scale_img, name)
    emb_array = obj.main_train()
    emb_list = list(map(list,emb_array))
    return jsonify({name: emb_list})

@app.route('/video', methods = ['POST'])
def video_feature():
  #value는 post한 json 정보 딕셔너리로 받는 코드 
    value = request.form.to_dict()
    
    value_str = value['images_placeholder']
  #value_str은 원본   
  #해당 과정을 통해서 받아온 데이터를 숫자만 남겨둔 상태. 
    value_st = value_str.replace(","," ")
    value_st = value_st.replace("["," ")
    value_st = value_st.replace("]"," ")
    
    value_st=value_st.split()
    #print(value_st)

  #value_st는 아예 float형태로 하나씩 저장함.  
    value_num =len(value_st)/3
    for i in range(0,len(value_st)):
      value_st[i] = float(value_st[i])

  #이차원 리스트 생성.
    value_list=[[0 for i in range(3)]for j in range(int(value_num))]
    for x in range(int(value_num)):
      value_list[x][0]=value_st[x*3]
      value_list[x][1]=value_st[x*3+1]
      value_list[x][2]=value_st[x*3+2]
    #print(value_st[0]) 
 
  #ndarray로 변경.
   # suppress는 소숫점 특정 자리 이후부터는 'e+~~' 이런식으로 출력하는 걸 다룸
   # precision은 소숫점 아래 몇자리까지 보여줄껀지 보여주는건데 그냥 넉넉히..20자리 했음... 
    np.set_printoptions(suppress=True,precision=20)
    #np.array가 데이터를 줄이고 있다.
    value_arr =np.array(value_list)
    print(value_arr)

    print(type(value_arr))
    
   
    sess = tf.Session()


    with sess.as_default():
        facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
    
    return jsonify(json_feed)
    
def template_Test():
   return render_template(
      'index.html',
      title = "Flask Template Test",
      my_str="Hello Flask!",
      my_list = [x +1 for x in range(30)]
      )


if __name__ =='__main__':
   app.run(debug=True)
   #서버 실행하는 부분. 