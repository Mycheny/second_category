# -*- coding: utf-8 -*- 
# @Time 2020/3/15 19:00
# @Author wcy
import os
import random

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from src.dataset import Dataset
from src.network import CostumeNetwork


def resize_image(image, width, height):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    if h / w < height / width:
        # "填充上下"
        hd = int(w * height / width - h + 0.5)
        top = hd // 2
        bottom = hd - top
        scale = w / width
    elif h / w > height / width:
        # "填充左右"
        wd = int(h * width / height - w + 0.5)
        left = wd // 2
        right = wd - left
        scale = h / height
    else:
        scale = 1
    # RGB颜色
    BLACK = [255, 255, 255]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (width, height)), [top / scale, bottom / scale, left / scale, right / scale]


if __name__ == '__main__':
    batch_size = 1
    src_shape = (128, 128)
    dataset = Dataset(src_shape=src_shape, src_path=None, batch_size=batch_size, shadow=0)

    model_path = "resources/model/train_0"
    root = "D:\\color\\images"
    net = CostumeNetwork(batch=batch_size, trainable=False)
    input = net.get_input()
    net_labels = net.labels
    logits = net.get_output()
    softmax = net.softmax
    count = 0
    ok = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        trainable_vars1 = tf.trainable_variables()
        TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
        trainable_vars2 = tf.get_collection(TRAINABLE_VARIABLES)
        freeze_conv_var_list = [t for t in trainable_vars1 if not t.name.startswith(u'conv')]

        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

        print(tf.train.latest_checkpoint(model_path))
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        datas = [[label, os.path.join(root, label, file_name)] for label in os.listdir(root) for file_name in
                 os.listdir(os.path.join(root, label))]
        random.shuffle(datas)
        for label, file_path in datas:
        #     # file_path = os.path.join("D:\\color\\images", "2", "1 (7).jpg")
        #     im = Image.open(file_path)
        #     frame = np.array(im)
        #     if len(frame.shape) < 3:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #     frame = frame[..., :3]
        #     # frame = cv2.imread(file_path)
        #     image = frame[..., ::1]
        #     image, edge = resize_image(image, 128, 128)
        #     # image_norm = image / 255
        #     image = image.astype(np.float)
        #     image_norm = (image - 127) / 127
        #     image_norm = np.expand_dims(image_norm, axis=0)
        #     data = dataset.next_batch_vali()
        #     image_norm1 = data['images_norm2']
        #     img = np.minimum(np.maximum((image_norm[0] + 1) / 2 * 255, 0), 255)
        #     img1 = np.minimum(np.maximum((image_norm1[0] + 1) / 2 * 255, 0), 255)
        #     cv2.imshow("", np.hstack((img.astype(np.uint8), img1.astype(np.uint8))))
        #     cv2.waitKey(100)
        #     result, res = sess.run([softmax, logits], feed_dict={input: image_norm})
        #     result1, res1 = sess.run([softmax, logits], feed_dict={input: image_norm1})
        #     if np.argmax(result, axis=1)[0] == int(label) - 1:
        #         ok += 1
        #     count += 1
        #     if count % 50 == 0: print(f"{ok}/{count}  {ok / count}")
        #     print(result, label)
        #     print(result1, label)
        #     # cv2.imshow("", image.astype(np.uint8))
        #     # cv2.waitKey(1)
        # print(f"{ok}/{count}  {ok / count}")

            # b = sess.run(bn_moving_vars[0])
            # print(f"{b}")
            data = dataset.next_batch_vali()
            images_NotNorm = data['images_NotNorm']
            image_norm = data['images_norm2']
            labels_index = data["labels_index"]
            labels = data["labels"]
            confusion_matrix = tf.confusion_matrix(tf.argmax(labels, 1), tf.argmax(softmax, 1), num_classes=2)
            result, res, acc, cm = sess.run([softmax, logits, net.accuracy, confusion_matrix], feed_dict={input: image_norm,
                                                                 net_labels: labels_index})
            if np.argmax(result, axis=1)[0]==labels_index[0]:
                ok+=1
            else:
                cv2.imwrite(f"resources/image82/{np.argmax(result, axis=1)[0]}_{labels_index[0]}_{str(ok)}.jpg", images_NotNorm[0].astype(np.uint8))
            count+=1
            print(cm)
            print(f"{ok}/{count} # {ok/count} # ", acc)

            # cv2.imshow("", images_NotNorm[0].astype(np.uint8))
            # cv2.waitKey(1)
            # print()

