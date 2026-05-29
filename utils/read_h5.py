
import os
import random


import numpy as np
import cv2
def read_data(path):

    # read file
    img_list = os.listdir(path + '\img_crop')

    random.shuffle(img_list)
    first_img = (cv2.cvtColor(cv2.imread(path + r'/img_crop/' + img_list[0]), cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5

    first_label = (cv2.cvtColor(cv2.imread(path + '/label_crop/' + img_list[0]), cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5


    for img in img_list[1:]:
        image = cv2.imread(path + '/img_crop/' + img)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5
        first_img = np.concatenate((first_img,image),0)

    for lab in img_list[1:]:
        label = cv2.imread(path + '/label_crop/' + lab)
        label = (cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5
        first_label = np.concatenate((first_label, label), 0)

    return first_img,first_label,img_list


def read_data_vi(path,index):

    # read file
    img_list = index

    random.shuffle(img_list)
    first_img = (cv2.cvtColor(cv2.imread(path + '/img_crop/' + img_list[0]), cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5

    first_label = (cv2.cvtColor(cv2.imread(path + '/label_crop/' + img_list[0]), cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5


    for img in img_list[1:]:
        image = cv2.imread(path + '/img_crop/' + img)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5
        first_img = np.concatenate((first_img,image),0)

    for lab in img_list[1:]:
        label = cv2.imread(path + '/label_crop/' + lab)
        label = (cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)[np.newaxis,:,:,np.newaxis]-127.5) / 127.5
        first_label = np.concatenate((first_label, label), 0)

    return first_img,first_label