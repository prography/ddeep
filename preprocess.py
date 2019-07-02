from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import detect_face

detector = mtcnn.MTCNN()
image_size = detector.image_size
input_image_size = detector.input_image_size

def collect_data(input_image):
    img = misc.imread(input_image)
    if img.ndim < 2:
        print("Unable !")
    elif img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0 : 3]
    
    minsize = 35  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 160
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')
    
                                                    
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet,
                                                onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
                                                                                                         
    if nrof_faces > 0:
        det = bounding_boxes[:, 0 : 4]
        img_size = np.asarray(img.shape)[0 : 2]

        bb_temp = np.zeros(4, dtype = np.int32)

        bb_temp[0] = det[0]
        bb_temp[1] = det[1]
        bb_temp[2] = det[2]
        bb_temp[3] = det[3]

        cropped_temp = img[bb_temp[1] : bb_temp[3],
                               bb_temp[0] : bb_temp[2], :]
        cropped_temp = facenet.flip(cropped_temp, False)
        scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp = 'bilinear')
        scaled_temp = cv2.resize(scaled_temp, (input_image_size, input_image_size),
                       interpolation = cv2.INTER_CUBIC)
        scaled_temp = facenet.prewhiten(scaled_temp)
        scaled_temp.reshape(-1, input_image_size, input_image_size, 3)

    return scaled_temp
