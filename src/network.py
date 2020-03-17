import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.contrib.slim import nets
from tensorflow.python.ops.rnn import dynamic_rnn

from src.network_base import BaseNetwork


class CostumeNetwork(BaseNetwork):
    def __init__(self, batch=1, image_width=128, image_height=128, keep_prob=1.0, trainable=True):
        self.batch, self.image_width, self.image_height = batch, image_width, image_height
        with tf.variable_scope(None, "input"):
            self.input = tf.placeholder(tf.float32, (batch, image_height, image_width, 3), name="input_image")
        inputs = {'image': self.input}
        self.labels = tf.placeholder(tf.int64, (batch), name="output_lable")
        self.keep_prob = keep_prob
        BaseNetwork.__init__(self, inputs, trainable)
        # if trainable:
        self.init_loss()

    def setup(self):
        with tf.variable_scope(None, "ExtractFeature"):
            feature_name = "feature"
            net, end_points = nets.resnet_v2.resnet_v2_50(self.input)
            feature = end_points[
                "{}/resnet_v2_50/block4/unit_3/bottleneck_v2/conv2".format(tf.get_variable_scope().name)]
            self.layers[feature_name] = feature
        with tf.variable_scope(None, "ClassifyFeature"):
            (self.feed(feature_name)
             .convb(3, 3, 256, 1, name="layer1")
             .convb(3, 3, 256, 1, name="layer2")
             .convb(3, 3, 128, 1, name="layer3")
             .convb(3, 3, 128, 1, name="layer4")
             .dropout(self.keep_prob, name="dropout")
             .reshape((self.batch, -1), name="feature")
             )
        with tf.variable_scope(None, "FullConnection"):
            (self.feed("feature")
             .fc(512, name="fc1")
             .fc(256, name="fc2")
             .fc(128, name="fc3")
             .fc(2, name="output", relu=False)
             )

    def get_input(self):
        return self.input

    def get_output(self):
        """
        获取所有输出，顺序[hand, heat, vect]
        :return:
        """
        return self.layers["output"]

    def init_loss(self):
        """
        初始化损失
        :return:
        """
        with tf.variable_scope(None, "losses"):
            # 目标函数
            logits = self.get_output()
            self.objective = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
            self.avg_loss = self.objective
            self.softmax = tf.nn.softmax(logits)
            correct_prediction = tf.equal(self.labels, tf.argmax(self.softmax, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            confusion_matrix = tf.confusion_matrix(self.labels, tf.argmax(self.softmax, 1), num_classes=2)
            TP = confusion_matrix[0, 0]
            FN = confusion_matrix[0, 1]
            FP = confusion_matrix[1, 0]
            TN = confusion_matrix[1, 1]
            self.confusion_accuracy = (TP+TN)/(TP+TN+FP+FN)  # 准确率
            self.confusion_error = (FP+FN)/(TP+TN+FP+FN)  # 错误率

            self.confusion_specificity = TN/(TN+FP)  # 特效度 表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力。
            self.confusion_precision = TP/(TP+FP)  # 精确率、精度  表示被分为正例的示例中实际为正例的比例。

            self.confusion_sensitive = TP / (TP + FN)  # 灵敏度 表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力。
            self.confusion_recall = TP/(TP+FN)  # 召回率 召回率与灵敏度是一样的。

            a = 1
            self.F1 = ((a**2+1)*self.confusion_precision*self.confusion_recall)/((self.confusion_precision+self.confusion_recall)*a**2)

            tf.summary.scalar("confusion_accuracy", self.confusion_accuracy)
            tf.summary.scalar("confusion_error", self.confusion_error)

            tf.summary.scalar("confusion_specificity", self.confusion_specificity)
            tf.summary.scalar("confusion_precision", self.confusion_precision)

            tf.summary.scalar("confusion_sensitive", self.confusion_sensitive)
            tf.summary.scalar("confusion_recall", self.confusion_recall)

            tf.summary.scalar("F1", self.F1)

            tf.summary.scalar("loss", self.avg_loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.image("image", self.input)


if __name__ == '__main__':
    net = CostumeNetwork()
