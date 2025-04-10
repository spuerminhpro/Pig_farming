import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''
shanghaiAtrain_path = '/mnt/sda1/PythonProject/Pig_counting/dataset/TransCrowd/train/images_crop/'
shanghaiAtest_path = '/mnt/sda1/PythonProject/Pig_counting/dataset/TransCrowd/test/images_crop/'

train_list = []
for filename in os.listdir(shanghaiAtrain_path):
    if os.path.splitext(filename)[1] == '.jpg':
        train_list.append(os.path.join(shanghaiAtrain_path, filename))

train_list.sort()
np.save('./npydata/Pig_detection_train.npy', train_list)

test_list = []
for filename in os.listdir(shanghaiAtest_path):
    if os.path.splitext(filename)[1] == '.jpg':
        test_list.append(os.path.join(shanghaiAtest_path, filename))
test_list.sort()
np.save('./npydata/Pig_detection_test.npy', test_list)

print("Generate pig detection image list successfully", len(train_list), len(test_list))