import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
from imutils import paths
import random
from cv2 import cv2
import os



#train_data_path = "/gitProject/dogs-vs-cats/train"

train_data_path = "./train/"
# X는 입력값
# Y는 출력값
#   cat : 0
#   dog : 1

#########

data = []
labels = []

image_paths = list(paths.list_images(train_data_path))
random.seed(42)
random.shuffle(image_paths)

print("[INFO] loading images...")
for file_name in image_paths:
    img = cv2.imread(file_name)
    # fx, fy : 리사이징 
    img = cv2.resize(img, dsize=(128, 128))
    data.append(img/256)
    splitted_file_name = file_name.split("\\")[-1]
    if "cat" in splitted_file_name:
        labels.append([1,0])
    elif "dog" in splitted_file_name:
        labels.append([0,1])


print("============================")

data = np.array(data)
labels = np.array(labels)

# 학습 데이터와 테스트 구분
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# 본격적으로 레이어 쌓는 부분
model = Sequential()

model.add(Convolution2D(16, (3, 3), padding='same', activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

##add layer - joonto
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#add layer 
model.add(Convolution2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#

model.add(Flatten())
model.add(Dense(32, activation='relu')) #modified 256->32
model.add(Dense(2, activation='sigmoid'))

# 대충 결과물파트
print("[INFO] start fitting.....")
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=32, epochs=15)

# 위에 학습한 모델 저장 
model.save('cnn_v1.h5')

# 테스트 데이터로 평가 진행
print("\n[INFO] evaluating........")
loss_and_metrics = model.evaluate(testX, testY, batch_size=32)

print('## evaluation loss and_metrics ##')
print(loss_and_metrics)





