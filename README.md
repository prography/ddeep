# Real Time Face Recognition using Facenet (실시간 영상 모자이크 처리 시스템)
### 기본 : Python  

1. Gui : tkinter  
2. 학습데이터 : CASIA-WebFace, VGGFace2 (triplet loss) / json으로 데이터 송수신  
3. 학습모델 : inception resnet  
4. 서버연동 : flask (AWS)  
5. 기술 : mtcnn (face detection), facenet (face recognition), cosine similarity (compare to feature map)  
6. 프레임워크 : tensorflow  

### 적용되지 않았지만 사용해왔던 것들  
1. Image Augmentation
+ 학습으로 사용할 이미지 데이터가 너무 많기 때문에 그 이미지에서 face detection 후 저장되는 Crop된 face image 또한 많다. 
+ 따라서 저장하고 가져오는데 소요되는 시간을 줄이기 위해서 1개의 이미지만을 input으로 넣고 augmentation해서 저장할 필요없이 바로 학습 데이터로 사용하기 위해 이용했다. 
+ 하지만, Cosine Similarity로 얼굴을 구분하는 형태로 바꼈기 때문에 Augmentation후 모델을 학습시킬 필요가 없어졌다.
2. Softmax, SVM (Support Vector Machine)
+ 마찬가지로 처음 facenet은 모델에 SVM으로 학습시킨 다음 예측할 때, Softmax로 classification 하는 형태였다. 
+ 하지만, fully connected layer 까지 들어갈 필요 없이 feature extraction하는 convolutional layers만 통과하고 난 뒤의,
+ output인 feature map을 이용하여 cosine similarity로 비교하는 형태로 바꼈기 때문에 변경하였다.
3. pytorch
+ 우리 수준에서 tensorflow보다는 코드가 짧고 직관적으로 알 수 있는 pytorch 코드가 더 낫다고 판단되어 중간에 변경하였다. 
+ 하지만, reference들에 한계가 있었고, 따라서 다시 tensorflow로 돌아왔다.
