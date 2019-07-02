import mtcnn
import cv2
import numpy as np
import tensorflow as tf
import facenet
from scipy import misc
import preprocess as prepro
import os
from classifier import training

modeldir = './model/20180402-114759.pb'
button_flag  = [1,1,1,1,1,1]
feature_list = []
button_name = ['','evans','hermsworth','jeremy','mark','olsen']


input_image="abc.jpg"
detector = mtcnn.MTCNN()
img = cv2.imread(input_image)

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
print('embedding_size:', embedding_size)


bounding_boxes, img = detector.run_mtcnn(img)
nrof_faces = bounding_boxes.shape[0]
if nrof_faces>0:
    det = bounding_boxes[:, 0:4]
    img_size = np.asarray(img.shape)[0:2]
    
    cropped = []
    scaled = []
    scaled_reshape = []
    bb = np.zeros((nrof_faces,4), dtype=np.int32)
    
    for i in range(nrof_faces):
        emb_array = np.zeros((1, embedding_size))
        
        bb[i][0] = det[i][0]
        bb[i][1] = det[i][1]
        bb[i][2] = det[i][2]
        bb[i][3] = det[i][3]
        
            
        cropped.append(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
        cropped[i] = facenet.flip(cropped[i], False)
        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                               interpolation=cv2.INTER_CUBIC)
        scaled[i] = facenet.prewhiten(scaled[i])
        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
        print('emb_array[0, :] : ', emb_array)
        img_data = facenet.check_features(feature_list, emb_array[0], {"name" : "", "cos_sim" : 0}, 0)
        print("name : ", img_data["name"], "\nsimilarity : ", img_data["cos_sim"])
        if img_data["cos_sim"] >= 0.6:
          
            if button_flag[button_name.index(img_data["name"])]%2 == 0:
                cv2.rectangle(img, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                
                
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                cv2.putText(img, img_data["name"], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 255), thickness=1, lineType=2)
            else:
                img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))
            
        else:                           
            img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]] = cv2.blur(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]], (23,23))

img = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
cv2.imshow('Video', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


scale_img = prepro.collect_data(os.path.join(os.getcwd(),'abc.jpg'))
obj = training(modeldir, scale_img, "evans")
get_feature = obj.main_train()
print('training image feature map emb_array: ', get_feature.feature)

