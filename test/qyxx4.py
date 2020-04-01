# -*- coding: utf-8 -*- 
# @Time 2020/3/24 15:36
# @Author wcy

#encoding=utf-8
# 迁移学习demo
import cv2
import glob
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 模型瓶颈层的节点个数
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

BOTTLENECK_TENSOR_SIZE = 2048

# 模型张量所对应的名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'                     # 瓶颈层
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'                  # 输入层

# MODEL_DIR = 'E:\\PycharmProjects\\款式分类\\test\\model_pb'           # 下载好的模型存放目录
MODEL_DIR = '/stor/wcy/costume_style/model_pb'           # 下载好的模型存放目录
MODEL_FILE = 'tensorflow_inception_graph.pb'                     # 模型文件

# CACHE_DIR = 'E:\\PycharmProjects\\款式分类\\test\\temp_bottleneck' # 图片经过瓶颈层之后输出的特征向量保存于文件之后的目录地址
# CACHE_DIR = 'D:\\2020.3.22款式图片\\temp_bottleneck' # 图片经过瓶颈层之后输出的特征向量保存于文件之后的目录地址
CACHE_DIR = '/stor/wcy/costume_style/temp_bottleneck' # 图片经过瓶颈层之后输出的特征向量保存于文件之后的目录地址

# INPUT_DATA = 'E:\\PycharmProjects\\款式分类\\test\\flower_photos'  # 训练图片文件夹
# INPUT_DATA = 'D:\\2020.3.22款式图片\\flower_photos'  # 训练图片文件夹
INPUT_DATA = '/stor/wcy/costume_style/flower_photos'  # 训练图片文件夹

VALIDATION_PERCENTAGE = 10     # 数据百分之10用于验证
TEST_PERCENTAGE = 10           # 数据百分之10用于测试, 剩下的百分之80用于训练

LEARNING_RATE = 0.01           # 学习率
STEPS = 40000                   # 训练迭代次数
BATCH = 256


def create_image_lists(testing_precentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取训练数据目录下的所有子文件夹
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:   # 得到的第一个目录是当前目录，不需要考虑
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpge', 'JPG', 'JPGE']
        file_list = []
        dir_name = os.path.basename(sub_dir)   # 返回最后一个'/'的目录名
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)  # 文件路径拼接
            file_list.extend(glob.glob(file_glob))   # 把所有符合要求的文件加入到file_lsit列表
        if not file_list : continue
        label_name = dir_name.lower()   # 大写转小写，花名
        training_images = []
        testing_images = []
        validation_images =[]
        for file_name in file_list:
            base_name = os.path.basename(file_name) # 图片名称
            # 随机将数据分到训练集, 测试集和验证集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            if chance < (testing_precentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 当前类别的数据放入结果字典
        result[label_name] = {
            'dir' : dir_name,
            'training' : training_images,
            'testing' : testing_images,
            'validation' : validation_images
        }
    # 返回整理好的数据
    return result

# 这个函数通过类别名称, 所属数据集和图片编码获取一张图片特征向量的地址
def get_image_path(image_lists, image_dir, label_name, index, category) :
    label_lists = image_lists[label_name]
    category_list = label_lists[category]   # category表示的是3种数据集的其中之一    training，testing，validation
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 这个函数通过类别名称，所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, lable_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, lable_name, index, category) + '.txt'   # 特征向量文件是.txt结尾的

# 这个函数使用加载训练好的Incepyion-v3 模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    try:
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)  # 压缩成一维的向量
        return bottleneck_values
    except Exception as e:
        print(end="")

# 这个函数获取一张图片经过Inception-v3模型处理之后的特征向量。这个函数会先试图寻找已经计算并且保存下来的特征向量，如果找不到则先计算这个特张向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_list = image_lists[label_name]
    sub_dir = label_list['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 特征向量文件不存在，则通过Inceptin-v3模型来计算特征向量，将计算的结果存入文件
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()   # 读取图片，准备输入到InceptionV3网络(没有改变图像大小)
        # 通过Inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        if bottleneck_values is None:
            print(image_path)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    # 否则直接从文件中获取图片相应的特征向量
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

# 这个函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        lable_index = random.randrange(n_classes)
        lable_name = list(image_lists.keys())[lable_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, lable_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[lable_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 这个函数获取全部的测试数据，在最终测试的时候需要在所有的测试数据上计算正确率(不是随机获取一个batch，而是全部数据)
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别的测试图片
    for label_index, label_name in enumerate(image_lists):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def draw_cm(cm):
    cell_w = 10
    h, w = cm.shape
    image = np.ones((h*cell_w, w*cell_w, 3), dtype=np.uint8)*255
    for i in range(h):
        cv2.line(image, (0, i * cell_w), (h*cell_w, i * cell_w), (0, 0, 0), lineType=cv2.LINE_AA)
        cv2.line(image, (i * cell_w, 0), (i * cell_w, w*cell_w), (0, 0, 0), lineType=cv2.LINE_AA)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ratio = cm[i,j]/cm.max()
            rgb = int(ratio*255)
            color = (rgb, rgb, rgb)
            cv2.rectangle(image, (i*cell_w+1, j*cell_w+1), (i*cell_w+cell_w-1, j*cell_w+cell_w-1), color, -1)
            # cv2.putText(image, f"{cm[i, j]}", (i*cell_w, j*cell_w), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imshow("image", image)
    cv2.waitKey(1)

def main():
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes =len(image_lists.keys())    # 5种花,该值为5
    # 载入已有的模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')  # 特征化过后的欲训练数据
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')                       # 对应的真实结果
    # 定义正向传播
    with tf.name_scope('final_training_ops'):
        weights1 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 1024], stddev=0.001))
        biases1 = tf.Variable(tf.zeros(1024))
        logits1 = tf.matmul(bottleneck_input, weights1) + biases1

        weights2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.001))
        biases2 = tf.Variable(tf.zeros(512))
        logits2 = tf.matmul(logits1, weights2) + biases2

        weights3 = tf.Variable(tf.truncated_normal([512, n_classes], stddev=0.001))
        biases3 = tf.Variable(tf.zeros(n_classes))
        logits = tf.matmul(logits2, weights3) + biases3

        final_tensor = tf.nn.softmax(logits)

    confusion_matrix = tf.confusion_matrix(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1), num_classes=n_classes)

    # 定义损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义反向传播
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("predict", evaluation_step)
    # 图定义结束
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("logs/qyxx4/train", sess.graph)
        vali_writer = tf.summary.FileWriter("logs/qyxx4/vali", sess.graph)
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            try:
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            except Exception as e:
                print("error")
                continue
            _, merged_summary_op_train = sess.run([train_step, merged_summary_op], feed_dict={bottleneck_input:train_bottlenecks, ground_truth_input: train_ground_truth})
            # 在验证数据上测试正确率
            if i%50 == 0 or i+1 == STEPS:
                try:
                    validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH*50, 'validation', jpeg_data_tensor, bottleneck_tensor)
                except Exception as e:
                    print("error vali")
                    continue
                validation_accuracy, cm, merged_summary_op_vali = sess.run([evaluation_step, confusion_matrix, merged_summary_op], feed_dict={bottleneck_input:validation_bottlenecks, ground_truth_input:validation_ground_truth})
                # draw_cm(cm)
                print(f"{i} validation_accuracy: {validation_accuracy}")
                train_writer.add_summary(merged_summary_op_train, i)
                vali_writer.add_summary(merged_summary_op_vali, i)

        # 在最后的测试数据上测试正确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input:test_bottlenecks, ground_truth_input:test_ground_truth})
        print(f"test_accuracy: {test_accuracy}")
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()