import tensorflow as tf
import numpy as np
import csv
def weight_variable(shape):
    tf.compat.v1.set_random_seed(1)
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))
# conv1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# full connection1
W_fc1 = weight_variable([10 * 10 * 64, 512])
b_fc1 = bias_variable([512])

W_fc2 = weight_variable([512, 14])
b_fc2 = bias_variable([14])


wconv1 = tf.zeros([5,5,1,32])
bconv1 = tf.zeros([32])
wconv2 = tf.zeros([5,5,32,64])
bconv2 = tf.zeros([64])
wfc1 = tf.zeros([10*10*64,512])
bfc1 = tf.zeros([512])
wfc2 = tf.zeros([512,14])
bfc2 = tf.zeros([14])

cpkt_model_path = "model"
# pb_model_path = "model//digital_gesture.pb"
param_path = 'model_params_001.csv'

# 从cpkt中加载参数
# saver = tf.compat.v1.train.Saver()
saver = tf.compat.v1.train.Saver(
    {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
     'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
with tf.compat.v1.Session() as sess:
    # 恢复模型
    module_file = tf.train.latest_checkpoint(cpkt_model_path)
    saver.restore(sess, module_file)

    # saver.restore(sess, cpkt_model_path)

    # 提取参数值
    wconv1 = sess.run(W_conv1)
    bconv1 = sess.run(b_conv1)
    wconv2 = sess.run(W_conv2)
    bconv2 = sess.run(b_conv2)
    wfc1 = sess.run(W_fc1)
    bfc1 = sess.run(b_fc1)
    wfc2 = sess.run(W_fc2)
    bfc2 = sess.run(b_fc2)


    # 保存到CSV文件
    with open(param_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入W_conv1权重
        writer.writerow("conv1 weight")
        for row in wconv1:
            writer.writerow(row)
        # 写入一个空行作为分隔
        writer.writerow([])
        writer.writerow("conv1 bias")
        # 写入b_conv1偏置
        writer.writerow(bconv1)

        # 写入W_conv1权重
        writer.writerow("conv2 weight")
        for row in wconv2:
            writer.writerow(row)
        # 写入一个空行作为分隔
        writer.writerow([])
        writer.writerow("conv1 bias")
        # 写入b_conv1偏置
        writer.writerow(bconv2)

        # 写入W_conv1权重
        writer.writerow("fc1 weight")
        for row in wfc1:
            writer.writerow(row)
        # 写入一个空行作为分隔
        writer.writerow([])
        writer.writerow("fc1 bias")
        # 写入b_conv1偏置
        writer.writerow(bfc1)

        # 写入W_conv1权重
        writer.writerow("fc2 weight")
        for row in wfc2:
            writer.writerow(row)
        # 写入一个空行作为分隔
        writer.writerow([])
        writer.writerow("fc2 bias")
        # 写入b_conv1偏置
        writer.writerow(bfc2)
