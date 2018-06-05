import os
import numpy as np
import tensorflow as tf

def weight_variable(shape,lambda1=0):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var=tf.Variable(initial)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')
def max_pool_2x2(x,strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=strides, padding='SAME')

def network1(train_data, keep_prob, lambda1):

    W_conv1 = weight_variable([11, 11, 1, 64], lambda1)
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(train_data, W_conv1, [1, 2, 2, 1]) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 64, 128], lambda1)
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv31 = weight_variable([3, 3, 128, 256], lambda1)
    b_conv31 = bias_variable([256])
    h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)
    # W_conv32 = weight_variable([5, 5, 384, 256])
    # b_conv32 = bias_variable([256])
    # h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)
    h_pool3 = max_pool_2x2(h_conv31)

    W_fc1 = weight_variable([16 * 16 * 256, 4096], lambda1)
    b_fc1 = bias_variable([4096])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([4096, 100], lambda1)
    b_fc2 = bias_variable([100])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv

def network2(train_data, keep_prob):

    W_conv11 = weight_variable([3, 3, 1, 32])
    b_conv11 = bias_variable([32])
    h_conv11 = tf.nn.relu(conv2d(train_data, W_conv11, [1, 2, 2, 1]) + b_conv11)
    W_conv12 = weight_variable([3, 3, 32, 32])
    b_conv12 = bias_variable([32])
    h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12,) + b_conv12)
    h_pool1 = max_pool_2x2(h_conv12)

    W_conv21 = weight_variable([3, 3, 32, 64])
    b_conv21 = bias_variable([64])
    h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
    W_conv22 = weight_variable([3, 3, 64, 64])
    b_conv22 = bias_variable([64])
    h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
    h_pool2 = max_pool_2x2(h_conv22)

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = bias_variable([128])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([16 * 16 * 128, 4096])
    b_fc1 = bias_variable([4096])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([4096, 100])
    b_fc2 = bias_variable([100])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def network3(train_data, keep_prob):

    W_conv11 = weight_variable([3, 3, 1, 32])
    b_conv11 = bias_variable([32])
    h_conv11 = tf.nn.relu(conv2d(train_data, W_conv11, [1, 2, 2, 1]) + b_conv11)
    W_conv12 = weight_variable([3, 3, 32, 32])
    b_conv12 = bias_variable([32])
    h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12,) + b_conv12)
    h_pool1 = max_pool_2x2(h_conv12)

    W_conv21 = weight_variable([3, 3, 32, 64])
    b_conv21 = bias_variable([64])
    h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
    W_conv22 = weight_variable([3, 3, 64, 64])
    b_conv22 = bias_variable([64])
    h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
    h_pool2 = max_pool_2x2(h_conv22)

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = bias_variable([128])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([8 * 8 * 256, 4096])
    b_fc1 = bias_variable([4096])
    h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([4096, 100])
    b_fc2 = bias_variable([100])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv