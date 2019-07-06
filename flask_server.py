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
    value = request.form.to_dict()
    print('------------------------------------------',type(value), '-------------')
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