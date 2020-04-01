import os
import random

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.network import CostumeNetwork


def main():
    net = CostumeNetwork(batch=1, trainable=True)
    input = net.get_input()
    net_labels = net.labels
    logits = net.get_output()
    softmax = net.softmax
    model_path = "resources/model/train_0"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        constant_ops = [op for op in sess.graph.get_operations()]
        for constant_op in constant_ops:
            print(constant_op.name)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                      ['input/input_image',
                                                                       'losses/Softmax'
                                                                       ])
        with tf.gfile.FastGFile('./resources/pb/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def resize_image(image, width, height):
    top, bottom, left, right = 0, 0, 0, 0
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


def get_image(path):
    im = Image.open(path)
    frame = np.array(im)
    if len(frame.shape) < 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = frame[..., :3]
    image = frame[..., ::1]
    image, edge = resize_image(image, 128, 128)
    image = image.astype(np.float)
    image_norm = (image - 127) / 127
    image_norm = np.expand_dims(image_norm, axis=0)
    return image_norm


def test():
    with tf.Session() as sess:
        with tf.gfile.FastGFile('./resources/pb/model.pb', 'rb') as f:
            graph_pen = tf.GraphDef()
            graph_pen.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_pen, name='')  # 导入计算图

            input = sess.graph.get_tensor_by_name('input/input_image:0')
            softmax = sess.graph.get_tensor_by_name('losses/Softmax:0')
            root = "D:\\color\\images"
            datas = [[label, os.path.join(root, label, file_name)] for label in os.listdir(root) for file_name in
                     os.listdir(os.path.join(root, label))]
            random.shuffle(datas)
            for label, file_path in datas:
                images = get_image(file_path)
                result = sess.run(softmax, feed_dict={input: images})
                print(result, label)


if __name__ == '__main__':
    # main()
    test()
