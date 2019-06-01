from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import os
import tensorflow as tf
import numpy as np
import detect_face
import facenet
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

def Augmentation(input_image, label):
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    aug_name = input_image.split("/")[-1].split(".")[0]
    
    minsize = 35  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 200
    
    nb_batches = 16
    
    aug_label = [label] * nb_batches
    aug_faces = []
    batches = []
    seq = iaa.Sequential([iaa.Fliplr(0.5),
                          sometimes(iaa.CropAndPad(
                                                   percent=(-0.05, 0.1),
                                                   pad_mode=ia.ALL,
                                                   pad_cval=(0, 255)
                                                   )),
                          sometimes(iaa.Affine(
                                               scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
                                               translate_percent={"x": (-0.2, 0.2), "y": (0, 0.2)},
                                               rotate=(-10, 10),
                                               shear=(-16, 16),
                                               order=[0, 1],
                                               cval=(0, 255)
                                               )),
                          iaa.SomeOf((0, 4),[
                                             iaa.OneOf([
                                                        iaa.GaussianBlur((0, 3.0)),
                                                        iaa.AverageBlur(k=(2, 7)),
                                                        iaa.MedianBlur(k=(3, 11)),
                                                        ]),
                                             
                                             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                                             iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)), # emboss images
                                             
                                             iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                                              iaa.EdgeDetect(alpha=(0.2, 0.5)),
                                                                              iaa.DirectedEdgeDetect(alpha=(0.2, 0.5), direction=(0.0, 1.0)),
                                                                              ])),
                                             
                                             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                                             
                                             iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                             
                                             iaa.Add((-10, 10), per_channel=0.5),
                                             
                                             iaa.AddToHueAndSaturation((-20, 20)),
                                             
                                             iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                                             
                                             iaa.Grayscale(alpha=(0.0, 1.0)),
                                             
                                             sometimes(iaa.ElasticTransformation(alpha=(0.5, 2), sigma=0.25)),
                                             
                                             sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
                                             
                                             sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                                             ], random_order=True)], random_order=True)
    img = misc.imread(input_image)
    if img.ndim < 2:
        print("Unable !")
    elif img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0 : 3]

    batches.append(np.array([img for _ in range(nb_batches)], dtype = np.uint8))

    aug_images = seq.augment_images(batches[0])
                                                     
    for aug_img in aug_images:
        bounding_boxes, _ = detect_face.detect_face(aug_img, minsize, pnet, rnet,
                                                    onet, threshold, factor)
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

            cropped_temp = aug_img[bb_temp[1] : bb_temp[3],
                                   bb_temp[0] : bb_temp[2], :]
            scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp = 'bilinear')
            aug_faces.append(scaled_temp)

    return (aug_label, aug_faces)

def Make_Data(root_dir):
    label_faces = []
    aug_faces = []
    names_dir = os.listdir(os.path.expanduser(root_dir))
    
    for i in range(len(names_dir)):
        middle_path = os.path.join(root_dir, names_dir[i])
        img_path = os.path.join(middle_path, os.listdir(middle_path)[0])
        
        label_temp, face_temp = Augmentation(img_path, i)
        
        label_faces.append(label_temp)
        aug_faces.append(face_temp)
    
    return (label_faces, aug_faces)
