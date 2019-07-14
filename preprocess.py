from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import mtcnn
import cv2


def collect_data(img):  
    detector = mtcnn.MTCNN()
    image_size = detector.image_size
    input_image_size = detector.input_image_size
                                                    
    bounding_boxes, img = detector.run_mtcnn(img)
    nrof_faces = bounding_boxes.shape[0]
                                                                                                         
    if nrof_faces > 0:
        det = bounding_boxes[:, 0 : 4]
        img_size = np.asarray(img.shape)[0 : 2]
        
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det = det[index, :]
                
        det = np.squeeze(det)
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
        scaled_reshape = scaled_temp.reshape(-1, input_image_size, input_image_size, 3)

    return scaled_reshape
