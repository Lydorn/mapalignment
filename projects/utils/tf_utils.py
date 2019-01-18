import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function
import math
import numpy as np


def get_tf_version():
    return tf.__version__


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def compute_current_adam_lr(optimizer):
    # print(get_tf_version())
    # a0, bb1, bb2 = optimizer._lr, optimizer._beta1_power, optimizer._beta2_power
    # at = a0 * (1 - bb2) ** 0.5 / (1 - bb1)
    # return at

    return optimizer._lr  # TODO: verify if this works


def count_number_trainable_params(trainable_variables=None):
    """
    Counts the number of trainable variables.
    """
    if trainable_variables is None:
        trainable_variables = tf.trainable_variables()
    tot_nb_params = 0
    for trainable_variable in trainable_variables:
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def get_nb_params_shape(shape):
    """
    Computes the total number of params for a given shape.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    """
    nb_params = 1
    for dim in shape:
        nb_params = nb_params * int(dim)
    return nb_params


def conv2d(x, W, stride=1, padding="SAME"):
    """conv2d returns a 2d convolution layer."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def complete_conv2d(input_tensor, output_channels, kernel_size, stride=1, padding="SAME", activation=tf.nn.relu, bias_init_value=0.025,
                    std_factor=1, weight_decay=None, summary=False):
    input_channels = input_tensor.get_shape().as_list()[-1]
    output_channels = int(output_channels)
    with tf.name_scope('W'):
        w_conv = weight_variable([kernel_size[0], kernel_size[1], input_channels, output_channels], std_factor=std_factor, wd=weight_decay)
        if summary:
            variable_summaries(w_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels], init_value=bias_init_value)
        if summary:
            variable_summaries(b_conv)
    z_conv = conv2d(input_tensor, w_conv, stride=stride, padding=padding) + b_conv
    if summary:
        tf.summary.histogram('pre_activations', z_conv)
    if activation is not None:
        h_conv = activation(z_conv)
    else:
        h_conv = z_conv
    if summary:
        tf.summary.histogram('activations', h_conv)
    return h_conv


def conv2d_transpose(x, W, output_shape, stride=1, padding="SAME"):  # TODO: add output_shape ?
    """conv2d_transpose returns a 2d transpose convolution layer."""
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)


def complete_conv2d_transpose(input_tensor, output_channels, output_size, kernel_size, stride=1, padding="SAME", activation=tf.nn.relu,
                              bias_init_value=0.025, std_factor=1, weight_decay=None, summary=False):
    batch_size = input_tensor.get_shape().as_list()[0]
    input_channels = input_tensor.get_shape().as_list()[-1]
    output_channels = int(output_channels)
    with tf.name_scope('W'):
        w_conv = weight_variable([kernel_size[0], kernel_size[1], output_channels, input_channels], std_factor=std_factor, wd=weight_decay)
        if summary:
            variable_summaries(w_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels], init_value=bias_init_value)
        if summary:
            variable_summaries(b_conv)
    z_conv = conv2d_transpose(input_tensor, w_conv, [batch_size, output_size[0], output_size[1], output_channels], stride=stride, padding=padding) + b_conv
    if summary:
        tf.summary.histogram('pre_activations', z_conv)
    h_conv = activation(z_conv)
    if summary:
        tf.summary.histogram('activations', h_conv)
    return h_conv


def complete_fc(input_tensor, output_channels, bias_init_value=0.025, weight_decay=None, activation=tf.nn.relu, summary=False):
    batch_size = input_tensor.get_shape().as_list()[0]
    net = tf.reshape(input_tensor, (batch_size, -1))
    input_channels = net.get_shape().as_list()[-1]
    with tf.name_scope('W'):
        w_fc = weight_variable([input_channels, output_channels], wd=weight_decay)
        if summary:
            variable_summaries(w_fc)
    with tf.name_scope('bias'):
        b_fc = bias_variable([output_channels], init_value=bias_init_value)
        if summary:
            variable_summaries(b_fc)
    z_fc = tf.matmul(net, w_fc) + b_fc
    if summary:
        tf.summary.histogram('pre_activations', z_fc)
    h_fc = activation(z_fc)
    if summary:
        tf.summary.histogram('activations', h_fc)
    return h_fc


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, std_factor=1, wd=None):
    """weight_variable generates a weight variable of a given shape. Adds weight decay if specified"""
    # Initialize using Xavier initializer
    fan_in = 100
    fan_out = 100
    if len(shape) == 4:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[3]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        print("WARNING: This shape format is not handled! len(shape) = {}".format(len(shape)))
    stddev = std_factor * math.sqrt(2 / (fan_in + fan_out))
    initial = tf.truncated_normal(shape, stddev=stddev)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial)


def bias_variable(shape, init_value=0.025):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial)


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    # with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def make_depthwise_kernel(a, in_channels):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    a = tf.constant(a, dtype=tf.float32)
    a = tf.tile(a, [1, 1, in_channels, 1])
    return a


def dilate(image, filter_size=2):
    rank = len(image.get_shape())
    if rank == 3:
        image = tf.expand_dims(image, axis=0)  # Add batch dim
    depth = image.get_shape().as_list()[-1]
    filter = np.zeros((filter_size, filter_size, depth))  # I don't know why filter with all zeros works...
    output = tf.nn.dilation2d(image, filter, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME", name='dilation2d')

    if rank == 3:
        return output[0]
    else:
        return output

    # rank = len(input.get_shape())
    # channels = input.get_shape().as_list()[-1]
    # kernel_size = 2*radius + 1
    # kernel_array = np.ones((kernel_size, kernel_size)) / (kernel_size*kernel_size)
    # kernel = make_depthwise_kernel(kernel_array, channels)
    # if rank == 3:
    #     input = tf.expand_dims(input, axis=0)  # Add batch dim
    # output = tf.nn.depthwise_conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    # if rank == 3:
    #     return output[0]
    # else:
    #     return output


def gaussian_blur(image, filter_size, mean, std):
    def make_gaussian_kernel(size: int,
                        mean: float,
                        std: float,
                        ):
        """Makes 2D gaussian Kernel for convolution."""
        mean = float(mean)
        std= float(std)
        d = tf.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

        gauss_kernel = tf.einsum('i,j->ij',
                                 vals,
                                 vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    gauss_kernel = make_gaussian_kernel(filter_size, mean, std)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    image_blurred = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    return image_blurred


def create_array_to_feed_placeholder(placeholder):
    shape = placeholder.get_shape().as_list()
    shape_removed_none = []
    for dim in shape:
        if dim is not None:
            shape_removed_none.append(dim)
        else:
            shape_removed_none.append(0)
    return np.empty(shape_removed_none)
