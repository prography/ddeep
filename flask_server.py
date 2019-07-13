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


class feature_map:
    def __init__(self, name, feature):
        self.name = name
        self.feature = feature

@app.route('/button/<name>')
def button_train(name):
    img = cv2.imread(os.path.join(os.getcwd(),'avengers/'+name+'/'+name+'_p.jpg'))
    scale_img = prepro.collect_data(img)
    obj = training(modeldir, scale_img, name)
    emb_array = obj.main_train()
    feature_list.append(feature_map(name, emb_array))
    
    return "Success!"

@app.route('/video', methods = ["POST"])
def video_feature():
    val = request.json
    value = val['images_placeholder']
    np.set_printoptions(suppress=True,precision=20)
    scaled_reshape = np.array(value)
  
    
    sess = tf.Session()
    with sess.as_default():
        facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    embedding_arr = np.zeros((1, embedding_size))
    
    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    embedding_arr[0, :] = sess.run(embeddings, feed_dict=feed_dict)
    
    img_data = facenet.check_features(feature_list, embedding_arr[0], {"name" : "", "cos_sim" : 0}, 0)
    print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])
    if img_data["cos_sim"] != 0:
        img_data["cos_sim"] = img_data["cos_sim"][0]
    return jsonify(img_data)
    
def template_Test():
   return render_template(
      'index.html',
      title = "Flask Template Test",
      my_str="Hello Flask!",
      my_list = [x +1 for x in range(30)]
      )


if __name__ =='__main__':
   app.run(host='0.0.0.0',port=5000,debug=True)
   #서버 실행하는 부분. 