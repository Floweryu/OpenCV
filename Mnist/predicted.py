'''
python 3.7
tensorflow 2.0.0
'''
import tensorflow as tf
from PIL import Image
import numpy as np

from train import CNN


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint(
            'D:/Learn_Files/TensorFlowL/Mnist/model/')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        # 训练集值的范围是[0, 1], 所以要除以255，不然1会识别成8
        flatten_img = np.reshape(img, (28, 28, 1)) / 255    
        x = np.array([1 - flatten_img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        # print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/0.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/1.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/2.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/3_3.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/4.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/5.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/6.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/7.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/8.png')
    app.predict('D:/Learn_Files/TensorFlowL/Mnist/test_data/9.png')
