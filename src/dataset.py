import copy
import json
import math
import os
import pickle
import platform
import random
import sys
import threading
import time
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageDraw


class Dataset(object):
    def __init__(self, src_shape=(128, 128), src_path=None, batch_size=3, channel=3, shadow=1):
        sysstr = platform.system()
        if sysstr == "Windows":
            if src_path is None:
                self.src_path = "D:\\color\\images"
            else:
                self.src_path = src_path
            self.cache_vali = 1  # 内存中缓存的验证batch数
            self.cache_train = 1  # 内存中缓存的训练batch数
            self.cache_max = 10  # 内存中缓存的最大训练batch数
        elif sysstr == "Linux":
            if src_path is None:
                self.src_path = "/stor/wcy/costume_data/images"
            else:
                self.src_path = src_path
            self.cache_vali = 5  # 内存中缓存的验证batch数
            self.cache_train = 30  # 内存中缓存的训练batch数
            self.cache_max = 120  # 内存中缓存的最大训练batch数
        else:
            print("Other System tasks")
            sys.exit(0)
        self.src_w = src_shape[0]
        self.src_h = src_shape[1]
        self.batch_size = batch_size
        self.channel = channel
        self.shadow = shadow
        data = {class_name:os.listdir(os.path.join(self.src_path, class_name)) for class_name in os.listdir(self.src_path)}
        self.class_num = len(list(data.keys()))
        self.class_name = list(data.keys())
        self.data = data
        random.seed(10)  # 设置随机种子，令每次打乱的都一样
        self.train_set = []
        self.vali_set = []
        self.init_data()
        self.production_vali = 4
        self.production_train = 4
        self.last_cons_vali = time.time()
        self.last_cons_train = time.time()

    def resize_image(self, image, width, height):
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

    def line_func(self, x1, y1, x2, y2, num=1):
        heat = np.zeros((self.tag_h, self.tag_w))
        if abs(x1 - x2) >= abs(y1 - y2):
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            x = np.arange(x1, x2 + 1)
            y = np.int64((x - x2) * (y1 - y2) / (0.000001 if (x1 - x2) == 0 else (x1 - x2)) + y2 + 0.5)
        elif abs(x1 - x2) < abs(y1 - y2):
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            y = np.arange(y1, y2 + 1)
            x = np.int64((y - y2) * (x1 - x2) / (0.000001 if (y1 - y2) == 0 else (y1 - y2)) + x2 + 0.5)
        else:
            x, y = int(x1), int(y1)
        x, y = np.maximum(np.minimum(x, self.tag_w * num - 1), 0), np.maximum(np.minimum(y, self.tag_h * num - 1), 0)
        heat[y, x] = 1
        return heat == 1

    def rotate(self, im, angle):
        """
        旋转图片
        :param im:
        :param lables: shape = [height，width, 5], 5的意义：[是否笔尖， 是否笔尾， x坐标占cell宽的百分比， y坐标占cell高度的百分比， 是否可见]
        :param angle: 选择角度
        :return:
        """
        color = (255, 255, 255)
        im_rotate = im.rotate(angle, fillcolor=color)
        return im_rotate

    def crop(self, image, tailor):
        """
        随机裁剪
        :param image:
        :param lables:
        :param tailor:
        :return:
        """
        width, height = image.size
        left = tailor
        right = width - tailor
        top = 0
        bottom = height
        image = image.crop((left, top, right, bottom))
        return image

    def noise(self, image):
        """
        添加一些随机的椒盐噪点
        :param image:
        :return:
        """

        def get_random_xy(size):
            x = random.randint(0, int(size[0]))
            y = random.randint(0, int(size[1]))
            return x, y

        def get_random_color():
            c1 = random.randint(0, 255)
            c2 = random.randint(0, 255)
            c3 = random.randint(0, 255)
            return c1, c2, c3

        if isinstance(image, Image.Image):
            draw = ImageDraw.Draw(image)
            for i in range(random.randint(0, int(image.size[0] * image.size[1] / 200))):
                draw.point([get_random_xy(image.size)], get_random_color())
            del draw
            image = np.array(image)
        dst = cv2.blur(image, (random.randint(1, 6), random.randint(1, 6)))
        # cv2.imshow("avg_blur_demo", dst)
        image = Image.fromarray(dst).convert('RGB')
        return image

    def center_gradient(self, shape, center=None):
        """
        返回以center为中心的渐变二维矩阵
        :param center: 中心点索引 [x, y]
        :param shape: 矩阵shape [w, h]
        :return: 二维矩阵
        """
        w, h = shape
        assert w > 0 and h > 0, "shape error: shape<0"
        if center is None:
            center = int(shape[0] / 2) - 1, int(shape[1] / 2) - 1
        x, y = center
        assert x < w and y < h, "Index out of range {}>{}".format(center, shape)
        row = np.tile(np.linspace(-x, -x + w - 1, w), [h, 1])
        col = np.tile(np.linspace(-y, -y + h - 1, h), [w, 1]).T
        image = np.sqrt(np.add(np.square(row), np.square(col)))
        image = image.max() - image
        if w > h:
            edge_value = image.max() - h / 2
        else:
            edge_value = image.max() - w / 2
        image = (image - edge_value) / (image.max() - edge_value)
        image[image < 0] = 0
        return image

    def generate_shadow(self, image, center, radius, brightness=0.8):
        """
        在image上生成随机的阴影
        :param image:
        :param center: 阴影中心
        :param radius: 阴影半径
        :param brightness: 阴影最小透明度
        :return:
        """
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.uint8)
        center = center[0] + radius, center[1] + radius
        point1, point2 = (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)
        shadow_h, shadow_w = point2[1] - point1[1], point2[0] - point1[0]
        shadow = 1 - self.center_gradient((shadow_w, shadow_h))
        shadow[shadow < brightness] = brightness
        shadow = np.tile(np.reshape(shadow, (shadow_h, shadow_w, 1)), [1, 1, image.shape[2]])
        left_w, left_h = point1
        right_w, right_h = point2
        image = cv2.copyMakeBorder(image, radius, 0, radius, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image_shadow = image[left_h:right_h, left_w:right_w, :]
        image_shadow = image_shadow * shadow[:image_shadow.shape[0], :image_shadow.shape[1], :]
        image[left_h:right_h, left_w:right_w, :] = image_shadow
        image = image[radius:, radius:, :]
        # image = Image.fromarray(image[radius:, radius:, :]).convert('RGB')
        return image

    def flip(self, image, args):
        index, angle, tailor = args

        if index < 2:
            image = image.transpose(index)
        elif index == 2:
            im_left_right = image.transpose(0)
            image = im_left_right.transpose(1)

        image = self.crop(image, tailor)
        image = self.rotate(image, angle)
        # 给图片添加随机噪点
        image = self.noise(image)
        image = np.array(image)
        # cv2.imshow("", np.hstack((image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))))
        # cv2.waitKey(0)
        return image

    def build_dataset(self, datas, data_type):
        assert len(datas.keys()) == self.class_num, "类别数量错误！"
        images = np.zeros((self.batch_size, self.src_h, self.src_w, self.channel))
        lables = np.zeros((self.batch_size, self.class_num))
        labels_index = []
        for i in range(self.batch_size):
            class_ = self.class_name[np.random.randint(0, len(self.class_name))]
            # class_ = "1"
            file_names = datas[class_]
            file_name = file_names[np.random.randint(0, len(file_names))]
            image_path = os.path.join(self.src_path, class_, file_name)
            # image_path = os.path.join(self.src_path, "2", "1 (7).jpg")

            flag = {"FLIP_LEFT_RIGHT": 0, "FLIP_TOP_BOTTOM": 1, "FLIP_TOP_BOTTOM_LEFT_RIGHT": 2, "NOTHING": 3}
            index = flag[random.sample(flag.keys(), 1)[0]]
            angle = random.sample(np.arange(-45, 45, 5).tolist(), 1)[0]
            tailor = random.randint(0, 20)
            args = [index, angle, tailor]
            frame = cv2.imread(image_path)
            if frame is None:
                im = Image.open(image_path)
            else:
                im = Image.fromarray(np.uint8(frame))
            if not data_type:
                image = self.flip(im, args)
            else:
                image = np.array(im)
            # image = np.array(im)
            image1, edge = self.resize_image(image, self.src_w, self.src_h)

            if self.shadow and random.random() < 0.2:
                h, w, _ = image1.shape
                brightness = min(max(random.random(), 0.1), 0.8)
                X, Y = random.randint(50, w - 50), random.randint(50, h - 50)
                X, Y = X + random.randint(0, 64), Y + random.randint(0, 64)
                radius = random.randint(32, max(w, h))
                image1 = self.generate_shadow(image1, (X, Y), radius, brightness=brightness)
            image1 = image1[:, :, ::-1]
            if self.channel == 1:
                image1 = np.expand_dims(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), axis=-1)
            else:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            images[i, :, :, :] = image1
            # print(class_)
            lables[i, self.class_name.index(class_)] = 1
            labels_index.append(self.class_name.index(class_))

        results = {"images_norm": images / 255,
                   "images_norm2": (images-127)/127,
                   "images_NotNorm": images,
                   "labels": lables,
                   "labels_index": labels_index,
                   }
        return results

    def next_baxtch(self):
        """
        获取下一batch
        :return:
        """
        self.consumption_train = max(time.time() - self.last_cons_train, 0.001)
        # （生产一次训练/验证集所花费的时间 - 上一次获取数据到现在的时间） 如果大于0 说明 产出速度小于消耗速度， 应调整缓存的数据数量
        if self.production_train - self.consumption_train > 0:
            multiple = int(self.production_train / self.consumption_train)
            cache = min(max(int(self.cache_train * multiple), self.cache_train), self.cache_max)
            self.cache_train = cache
            # print("cache_train = {} {}".format(self.cache_train, multiple))
        else:
            # 产出速度大于消耗速度
            self.cache_train = 2
            # print("cache_train = {}".format(self.cache_train))
        while True:
            time.sleep(0.1)
            if len(self.train_set) > 0:
                res = self.train_set.pop(0)
                self.last_cons_train = time.time()
                return res

    def next_batch_vali(self):
        """
        获取下一batch 验证数据
        :return:
        """
        self.consumption_vali = max(time.time() - self.last_cons_vali, 0.001)
        # （生产一次训练/验证集所花费的时间 - 上一次获取数据到现在的时间） 如果大于0 说明 产出速度小于消耗速度， 应调整缓存的数据数量
        if self.production_vali - self.consumption_vali > 0:
            multiple = int(self.production_vali / self.consumption_vali)
            cache = min(max(int(self.cache_vali * multiple), self.cache_vali), self.cache_max)
            self.cache_vali = cache
            # print("cache_vali = {} {}".format(self.cache_vali, multiple))
        else:
            # 产出速度大于消耗速度
            self.cache_vali = 2
            # print("cache_vali = {}".format(self.cache_vali))
        while True:
            time.sleep(0.1)
            if len(self.vali_set) > 0:
                res = self.vali_set.pop(0)
                self.last_cons_vali = time.time()
                return res

    def load_data(self, datas, data_type=True):
        start = time.time()
        results = self.build_dataset(datas, data_type)
        if data_type:
            self.vali_set.append(results)
            # print("\r dataset join vali_set [{}, {}, {}]".format(len(self.vali_set), self.cache_vali,
            #                                                      len(threading.enumerate())), end="")
            sys.stdout.flush()
            self.production_vali = time.time() - start
        else:
            self.train_set.append(results)
            # print("\r dataset join train_set [{}, {}, {}]".format(len(self.train_set), self.cache_train,
            #                                                       len(threading.enumerate())), end="")
            sys.stdout.flush()
            self.production_train = time.time() - start

    def load_train_thread(self, DATAS, data_type):
        """
        判断生成的数据集是否足够，不够再开线程进行生产
        :param datas:
        :param data_type: 数据类型  若是 True 则是加入验证集，反之加入训练集
        :return:
        """
        while True:
            keys = list(DATAS.keys())
            while True:
                time.sleep(0.1)
                if data_type:
                    if len(self.vali_set) < self.cache_vali and len(threading.enumerate()) < self.cache_max:
                        create = self.cache_vali - len(self.vali_set)
                        # print("create vail {}".format(create))
                        break
                else:
                    if len(self.train_set) < self.cache_train and len(threading.enumerate()) < self.cache_max:
                        create = self.cache_train - len(self.train_set)
                        # print("create train {}".format(create))
                        break
            threads = []
            for i in range(create):
                t = threading.Thread(target=self.load_data, args=(DATAS, data_type))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

    def init_data(self):
        """
        初始化数据，将数据拆分为训练集和验证集,并分别送入两个子线程中处理
        :return: None
        """
        self.vali_data = dict()
        self.train_data = dict()
        for key, value in self.data.items():
            border = int(0.9*len(value))
            self.train_data[key] = value[:border]
            self.vali_data[key] = value[border:]
        # print("\n######### 加载验证集 ########")
        threading.Thread(target=self.load_train_thread, args=(self.vali_data, True)).start()
        # print("\n######### 加载训练集 ########")
        threading.Thread(target=self.load_train_thread, args=(self.train_data, False)).start()


if __name__ == '__main__':
    src_shape = (128, 128)
    dataset = Dataset(src_shape=src_shape, src_path=None, batch_size=8, shadow=1)
    i = 0
    # with tf.Session() as session:
    while True:
        if i % 2 == 0:
            data = dataset.next_baxtch()
            print("训练数据")
        else:
            data = dataset.next_batch_vali()
            print("验证数据")
        i += 1
        images = data['images_NotNorm']
        labels = data["labels"]
        for j, (image, label) in enumerate(zip(images, labels)):
            print(label, data["labels_index"][j])
            cv2.imshow("", image.astype(np.uint8))
            cv2.waitKey(100)
