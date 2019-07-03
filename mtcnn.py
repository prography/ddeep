import tensorflow as tf
import cv2
import facenet
import detect_face
import numpy as np

npy='./npy'

class MTCNN:
    def __init__(self):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.session.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.session, npy)
                
                self.minsize=35
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor
                self.margin = 44
                self.frame_interval = 3
                self.image_size = 182
                self.input_image_size = 160        
        
    def run_mtcnn(self, img):
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
       
        return bounding_boxes, img
                
