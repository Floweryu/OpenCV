import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

'''
定义神经网络层函数，主要用于隐藏层和输出层
inputs代表输入的数据，  inSize为数据的形状  outSize为隐藏增的神经元个数，也为输出数据的形状
activationFunction 激活函数，默认无激活函数
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))       # 参数权重
    biases = tf.Variable(tf.zeros([1, out_size]))       # 偏量值
    wx_plus_b = tf.matmul(inputs, weights) + biases     # y = wx + b
    # 判断激活函数
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs
    

# get datas of sin
def get_sin_data():
    x_data = np.linspace(-math.pi, math.pi, 300) [:, np.newaxis]       # get 300 x
    # according normal to get noises of the same size as x
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.sin(x_data) + noise                     # add noise to y
    return x_data, y_data


def main():
    '''
    用placeholder进行数据输入的占位
    xs代表x_data特征集， ys代表y_data标签集
    [None, 1]其中的None表示可以接受任意数据，比如[300, 1], [200, 1]均可
    '''
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    '''
    https://blog.csdn.net/u013555719/article/details/77863953
    构造的神经网络比较简单，  输入层->隐藏层->输出层，其中输入层没有经过任何数据的处理，直接传递给隐藏层
    '''
    # 添加隐藏层
    layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)      # "tf.nn.relu" is max(features, 0)
    
    prediction = add_layer(layer1, 10, 1, activation_function=tf.nn.tanh)   # 添加输出层

    loss = tf.reduce_mean(tf.square(ys - prediction))   # 损失函数，计算loss得分
 
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss) # 梯度下降，使得分变少

    init = tf.global_variables_initializer()    # 初始化所有变量

    '''
    运行测试
    '''
    sess = tf.Session()
    sess.run(init)

    # plot the real data
    x_data, y_data = get_sin_data()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()       # 打开动态连续画图
    plt.show()

    for i in range(5000):
        # 使用feed_dict传递数据，训练神经网络
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # 每训练50次打印一次损失函数得分
            print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            # 训练输出层
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)


if __name__ == "__main__":
    main()

