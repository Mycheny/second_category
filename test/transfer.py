# -*- coding: utf-8 -*- 
# @Time 2020/3/14 20:37
# @Author wcy
import os
import time

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops

from src.dataset import Dataset
from src.network import CostumeNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])


def get_freeze_tensor():
    with tf.gfile.FastGFile("model_pb/tensorflow_inception_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input0, output0 = tf.import_graph_def(graph_def, return_elements=["DecodeJpeg/contents:0", "pool_3/_reshape:0"])
        return input0, output0


def build_net():
    input0, output0 = get_freeze_tensor()
    print()

def train():
    batch_size = 64
    lr = 0.001
    logpath = "resources/logs/"
    model_path = "resources/model"
    pre_model = "pre_model"
    train_name = f"train_{gpu_id}"
    test_name = f"test_{gpu_id}"

    build_net()
    # 定义优化器
    step_per_epoch = 50  # 切换学习率间隔步数
    global_step = tf.Variable(0, trainable=False)
    # 学习率连续衰减
    starter_learning_rate = lr
    end_learning_rate = 0.000005
    decay_rate = 0.01
    start_decay_step = 100
    decay_steps = 5000  #
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),  # if global_step >= start_decay_step
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step, decay_steps,
                                      end_learning_rate, power=1.0),
            starter_learning_rate
        )
    )
    tf.summary.scalar("learning_rate", learning_rate)
    with tf.variable_scope(None, "optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)

    trainable_vars1 = tf.trainable_variables()
    TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
    trainable_vars2 = tf.get_collection(TRAINABLE_VARIABLES)
    freeze_conv_var_list = [t for t in trainable_vars1 if not t.name.startswith(u'conv')]
    tf.summary.histogram("trainable_vars1", trainable_vars1[124])
    merged_summary_op = tf.summary.merge_all()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    # saver1 = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        # get_freeze(sess)
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(logpath + train_name, sess.graph)
        test_writer = tf.summary.FileWriter(logpath + test_name, sess.graph)

        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        # 重置 global_step 为 0
        sess.run(tf.assign(global_step, 0))
        initial_gs_num = sess.run(global_step)
        print("initial_gs_num={}".format(initial_gs_num))
        dataset = Dataset(batch_size=batch_size)
        max_epoch = 99999999
        while True:
            train_set = dataset.next_baxtch()
            images_train = train_set['images_norm2']
            lables_train = train_set['labels_index']
            _, gs_num, b = sess.run([train_op, global_step, bn_moving_vars[0]],
                                 feed_dict={input: images_train,
                                            label: lables_train
                                            })
            if gs_num > step_per_epoch * max_epoch:
                break

            if gs_num - last_gs_num >= 5:
                # print(b)
                train_loss_total, train_accuracy, train_lr_val, train_summary = sess.run(
                    [total_loss, net.accuracy, learning_rate,
                     merged_summary_op],
                    feed_dict={input: images_train,
                               label: lables_train
                               },
                )
                test_set = dataset.next_batch_vali()
                images_test = test_set['images_norm2']
                lables_test = test_set['labels_index']
                test_loss_total, test_accuracy, test_lr_val, test_summary = sess.run(
                    [total_loss, net.accuracy, learning_rate, merged_summary_op],
                    feed_dict={input: images_test,
                               label: lables_test,
                               },
                )
                print(f"train_loss_total: {train_loss_total} test_loss_total: {test_loss_total}\ntrain_accuracy: {train_accuracy} test_accuracy: {test_accuracy}")
                last_gs_num = gs_num
                if gs_num >= 0:
                    # train_writer.add_run_metadata(run_metadata, 'step%03d' % gs_num, global_step=gs_num)
                    # test_writer.add_run_metadata(run_metadata, 'step%03d' % gs_num, global_step=gs_num)
                    train_writer.add_summary(train_summary, gs_num)
                    test_writer.add_summary(test_summary, gs_num)

            if gs_num - last_gs_num2 >= 100:
                saver.save(sess, os.path.join(model_path, train_name, 'model'),
                           global_step=global_step)
                last_gs_num2 = gs_num


if __name__ == '__main__':
    train()