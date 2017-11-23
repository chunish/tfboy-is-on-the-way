# topic：    使用单层MLP实现MNIST 字符识别
# 主要改进：  防止过拟合——dropout；防止参数难调——Adagrad；防止梯度弥散——ReLu作为激活函数
# 结果：      相比与softmax回归的92%左右，单层MLP可达到接近98%
#
import tensorflow as tf
#import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print("MNIST ready")
sess = tf.InteractiveSession()

in_units = 784  #输入节点数
h1_units = 300  #隐含层的输出节点数

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10])) 



x = tf.placeholder(tf.float32, [None, in_units]) 
keep_prob = tf.placeholder(tf.float32) 

# 使用ReLu作为激活函数，不用sigmoid或者tanh等避免梯度弥散
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# 使用dropout解决过拟合问题
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# 选用softmax作为输出层函数，因其更符合概率分布
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10]) 
#使用交叉熵作为损失函数
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
# 使用Adagrad作为优化函数，自适应调整训练速度
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 初始化所有参数
tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})


# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # 对比预测值的索引和真实label的索引是否一样，一样返回True，不一样返回False
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print("FUNCTIONS READY")

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print("DONE")
