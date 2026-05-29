import glob
import os

import cv2
import h5py


def imread(path):
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[:, :]


def imread_gray(path):
    img = cv2.imread(path) / 255
    return img[:, :, 0]


def input_setup(config, data_dir, ir_flag, index=0):

    # Load data path
    if config.is_train:
        # 取到所有的原始图片的地址
        data = prepare_data(config, dataset=data_dir)
    else:
        data = prepare_data(config, dataset=data_dir)

    padding = abs(config.image_size - config.label_size) / 2  # 6

    for i in range(len(data)):

        input_ = imread(data[i])

        label_ = input_

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        img_count = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                img_count += 1
                sub_input = input_[int(x):int(x + config.image_size),
                            int(y):int(y + config.image_size)]
                sub_label = label_[int(x + padding):int(x + padding + config.label_size),
                            int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]
                # Make channel value
                if data_dir == "Train":
                    sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                                           interpolation=cv2.INTER_CUBIC)
                    sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                    sub_label = cv2.resize(sub_label, (config.label_size / 4, config.label_size / 4),
                                           interpolation=cv2.INTER_CUBIC)
                    sub_label = sub_label.reshape([config.label_size / 4, config.label_size / 4, 1])
                else:
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

                save(img_count, sub_input, 'img',ir_flag)
                save(img_count, sub_label, 'label',ir_flag)


def make_data(config, data, label, data_dir):

    if config.is_train:
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    else:
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'test.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def prepare_data(config, dataset):

    if config.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        # 将图片按序号排序
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    print(data)

    return data


import numpy as np
from PIL import Image


def save(count,img,flag,ir_flag):
    image = Image.fromarray(np.uint8(img * 255))
    if ir_flag:
        if flag == 'label':
            image.save(r'E:\begoina\fir-gan\dataset\vehicle\ir\label_crop/' + str(count) + '.png')
            print('success'+str(count))
        elif flag == 'img':
            image.save(r'E:\begoina\fir-gan\dataset\vehicle\ir\img_crop/'+ str(count) + '.png')
            print('success'+str(count))
        else:
            raise SystemExit(0)
    else:
        if flag == 'label':
            image.save(r'E:\begoina\fir-gan\dataset\vehicle\vis\label_crop/' + str(count) + '.png')
            print('success'+str(count))
        elif flag == 'img':
            image.save(r'E:\begoina\fir-gan\dataset\vehicle\vis\img_crop/'+ str(count) + '.png')
            print('success'+str(count))
        else:
            raise SystemExit(0)