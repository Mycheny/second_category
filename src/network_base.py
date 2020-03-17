import random
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import src.common as common

DEFAULT_PADDING = 'SAME'

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.1)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_l2_regularizer_convb = tf.contrib.layers.l2_regularizer(common.regularizer_conv)


def layer(op):
    """
    可组合网络层的装饰器
    :param op:
    :return:
    """

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name',
                                 self.get_unique_name(op.__name__))  # get() 方法类似, 如果键不存在于字典中，将会添加键并将值设为默认值(首字母会变大写)。
        if len(self.terminals) == 0:
            raise RuntimeError('还没有定义输入层 %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # layer_output = tf.clip_by_value(layer_output, -10000, 10000)
        tf.summary.histogram("{0}_output".format(name), layer_output)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class BaseNetwork(object):
    def __init__(self, inputs, trainable=True):
        # 这个网络的输入节点
        self.inputs = inputs
        # The current list of terminal nodes
        # 终端节点的当前列表
        self.terminals = []
        # Mapping from layer names to layers
        # 从层名到层的映射
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        # 如果为真，则将结果变量设置为可训练变量
        self.trainable = trainable
        with tf.variable_scope(None, "Network"):
            self.setup()

    def setup(self):
        '''Construct the network.构建网络 '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='bytes').item()
        for op_name in data_dict:
            if isinstance(data_dict[op_name], np.ndarray):
                if 'RMSProp' in op_name:
                    continue
                with tf.variable_scope('', reuse=True):
                    var = tf.get_variable(op_name.replace(':0', ''))
                    try:
                        session.run(var.assign(data_dict[op_name]))
                    except Exception as e:
                        print(op_name)
                        print(e)
                        sys.exit(-1)
            else:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        try:
                            var = tf.get_variable(param_name.decode("utf-8"))
                            session.run(var.assign(data))
                        except ValueError as e:
                            print(e)
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            try:
                is_str = isinstance(fed_layer, basestring)
            except NameError:
                is_str = isinstance(fed_layer, str)
            if is_str:
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_layer(self, name=None):
        '''Returns the current network output.'''
        if not name:
            return self.terminals[-1]
        else:
            return self.layers[name]

    def get_unique_name(self, prefix):
        """
        返回给定前缀的索引后缀唯一名称。
        它用于基于类型前缀自动生成层名称。
        :param prefix:
        :return:
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, trainable=True):
        '''创建新的tensor变量'''
        return tf.get_variable(name, shape, trainable=self.trainable & trainable,
                               initializer=tf.contrib.layers.xavier_initializer())

    def validate_padding(self, padding):
        '''验证填充参数是否有效'''
        assert padding in ('SAME', 'VALID')

    @layer
    def normalize_vgg(self, input, name):
        # normalize input -1.0 ~ 1.0
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_mobilenet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_nasnet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def input_pad(self, input, name):
        """
        未完成
        :param input:
        :param name:
        :return:
        """
        shape = input.get_shape()
        shape_list = shape.as_list()
        paddings = tf.constant([[0, 0], [10, 0], [0, 20], [0, 0]])
        input1 = tf.pad(input, paddings, name=name)
        return

    @layer
    def normalize_input(self, input, name):
        """
        如果输入时3维则将图片归一化，是4维则不处理
        :param input:
        :param name:
        :return:
        """
        shape = input.get_shape().as_list()
        if len(shape) < 4:
            input = tf.divide(input, 255.0, name=name + '_divide')
            input = tf.reshape(input, (1, shape[0], shape[1], shape[2]))
        return input

    @layer
    def upsample(self, input, factor, name="resize_{0}".format(str(random.randint(0, 100)))):
        """
        使用双线性插值将input的尺寸放大factor倍
        :param input: 4维张量  shape [batch, height, width, channels]
        :param factor: 放大倍数  type int or list
        :param name: 名称
        :return:
        """
        if not isinstance(factor, list):
            factor = [factor, factor]
        return tf.image.resize_bilinear(input, [int(input.get_shape()[1].value * factor[0]), int(input.get_shape()[2].value * factor[1])],
                                        name=name)



    @layer
    def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True):
        """
        实现深度可分离卷积
        :param input:
        :param k_h: 卷积核高度
        :param k_w: 卷积核宽度
        :param c_o: 输出通道数
        :param stride: 步长
        :param name: 操作名
        :param relu: 是否使用relu激活函数
        :param set_bias: 是否加上偏置
        :return:
        """
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=self.trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_h, k_w],
                                                  # activation_fn=common.activation_fn if relu else None,
                                                  activation_fn=None,
                                                  normalizer_fn=slim.batch_norm,
                                                  # weights_initializer=_init_norm,
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding=DEFAULT_PADDING,
                                                  scope=name + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=common.activation_fn if relu else None,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=_init_norm,
                                        biases_initializer=_init_zero if set_bias else None,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=self.trainable,
                                        weights_regularizer=None,
                                        scope=name + '_pointwise')
        frature = tf.transpose(output, [3, 1, 2, 0])[..., 0:1]
        tf.summary.image(name, frature, max_outputs=32)
        return output

    @layer
    def convolution2d_transpose(self, inputs, num_outputs, name, kernel_size=(3, 3), stride=2):
        output = slim.convolution2d_transpose(inputs,
                                              num_outputs,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=DEFAULT_PADDING,
                                              scope=name)
        return output

    @layer
    def convb(self, input, k_h, k_w, num_outputs, stride, name, relu=True, set_bias=True, set_tanh=False):
        # arg_scope 用于为tensorflow里的layer函数提供默认值
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.convolution2d(input,
                                        num_outputs,
                                        kernel_size=[k_h, k_w],
                                        stride=stride,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=_l2_regularizer_convb,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        biases_initializer=_init_zero if set_bias else None,
                                        trainable=self.trainable,
                                        activation_fn=common.activation_fn if relu else None,
                                        scope=name)
            if set_tanh:
                output = tf.nn.sigmoid(output, name=name + '_extra_acv')
        return output

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             trainable=True,
             biased=True):
        self.validate_padding(padding)
        c_i = int(input.get_shape()[-1])
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=self.trainable & trainable)
            if group == 1:
                output = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                output = tf.concat(3, output_groups)
            if biased:
                biases = self.make_var('biases', [c_o], trainable=self.trainable & trainable)
                output = tf.nn.bias_add(output, biases)

            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def hard_sigmoid(self, input, max_value=2, name="hard_sigmoid"):
        ReLU6 = tf.keras.layers.ReLU(max_value=max_value, name=name)
        output = ReLU6(input)
        return output


    @layer
    def sigmoid(self, input, name, interval=None):
        """
        将输出的结果限制在区间 interval (默认[0,1])之间，总和大于1
        :param interval: 压缩区间
        :param input:
        :param name:
        :return:
        """
        output = tf.nn.sigmoid(input, name)
        if interval is None:
            return output
        size = interval[1] - interval[0]
        negative = -interval[0]
        output = tf.multiply(output, size)
        output = tf.subtract(output, negative)
        return output

    @layer
    def tanh(self, input, name):
        """
        将输出的结果限制在0-1之间
        :param input:
        :param name:
        :return:
        """
        return tf.nn.tanh(input, name=name)

    @layer
    def softmax(self, input, name):
        """
        将输出的结果限制在0-1之间，总和等于1
        :param input:
        :param name:
        :return:
        """
        return tf.nn.softmax(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def reshape(self, inputs, shape, name):
        return tf.reshape(inputs, shape, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        shape_in = self.layers["image"].get_shape().as_list()
        shape_out = input.get_shape().as_list()
        if len(shape_in) < 4:
            input = tf.reshape(input, (shape_out[1], shape_out[2], shape_out[3]))
        return tf.nn.dropout(input, keep_prob, name=name)


