import math
from python.brisquequality import *

def get_entropy(img_):
    x, y = img_.shape[0:2]
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


def qem(ir_tensor_batch, vi_tensor_batch,d_train_number,index_,batch_size):


    b, c, h, w = ir_tensor_batch.shape
    qem_sum = 0
    for i in range(b):
        ir_temp = ir_tensor_batch[i, 0, ..., ...]
        vi_temp = vi_tensor_batch[i, 0, ..., ...]

        index_now = index_[i]

        ir_np = ir_temp.cpu().numpy() * 127.5 + 127.5
        vi_np = vi_temp.cpu().numpy() * 127.5 + 127.5

        ir_np1 = ir_np.astype(int)
        res_ir = get_entropy(ir_np1)
        E1 = res_ir / 8

        vi_np1 = vi_np.astype(int)
        res_vi = get_entropy(vi_np1) + 0.01
        E2 = res_vi / 8


        # read txt
        with open(r'E:\begoina\firGan\brisque_v\brisque_v' + str(index_now[:-4]) + '.txt','r',encoding='utf-8') as file:
            lines = file.readlines()
            Q_ir = float(lines[0].split(',')[0])
            Q_vi = float(lines[0].split(',')[1])
        ################################################################################################################
        # # brisque生成代码
        # if not test_img(ir_np) or not test_img(vi_np):
        #     qem_tmp = E2 / E2 + E1
        #     Q_ir = 1
        #     Q_vi = 1
        # else:
        #     Q_ir = test_measure_BRISQUE1(ir_np)
        #     Q_vi = test_measure_BRISQUE1(vi_np)
        #     qem_tmp = (Q_vi * E2) / (Q_vi * E2 + Q_ir * E1)
        #
        #
        # # brisque写入txt
        # if d_train_number == 0:
        #     text = str(Q_ir) + ',' + str(Q_vi)
        #     with open(r'E:\begoina\fir-gan\brisque_v' + str(index_now[:-4]) + '.txt','w',encoding='utf-8') as file:
        #         file.write(text)
        #     import os
        #
        #
        #     save_dir = r'E:\begoina\fir-gan\brisque_v'  # 你可以改这里
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)  # 自动创建文件夹
        #
        #
        #     file_name = os.path.basename(index_now).rsplit('.', 1)[0]  # 去掉后缀
        #     save_path = os.path.join(save_dir, f'brisque_v{file_name}.txt')
        #
        #
        #     with open(save_path, 'w', encoding='utf-8') as file:
        #         file.write(text)
        #     #############################################################################################################
        qem_tmp = (Q_vi * E2) / (Q_vi * E2 + Q_ir * E1)
        qem_sum += qem_tmp

    qem = qem_sum / b

    return qem