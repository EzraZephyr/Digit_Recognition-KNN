import warnings   # 忽略警告信息
import joblib   # 加载和保存训练好的模型
import numpy as np  # 操作数组
from PIL import Image  # 处理图像文件

class DigitRecognizer:
    # 创建一个手写数字识别器的类

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        # 使用joblib方法加载训练好的模型
        self.X_model = self.model._fit_X
        self.y_model = self.model._y
        # 取出模型的训练数据和标签

    def compute_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        # 通过公式计算两个数之间的欧式距离

    def compute_weight(self, distance):
        a, b = 1, 1
        # 定义用来调整权重的参数 a可以理解为一个平滑引子 用来避免发生除以0的情况
        return b / (distance + a)
        # 计算并返回权重值 这个公式通过带入不同的距离可以发现 距离越小 权重越大 距离越大 权重越小

    def predict_digit(self, filename):
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, -1)
        # 处理传入进来的图像 之前解释过了就不再说一遍了

        distances = []
        # 创建一个空列表 用于储存每个训练样本与输入图像的距离和标签
        for i, X_train in enumerate(self.X_model):
            # 遍历训练集中的每一个样本

            distance = self.compute_distance(img, X_train.reshape(1, -1))
            # 将图像和训练样本投入到compute_distance函数中进行计算 别忘了转换训练样本以匹配imn的形状

            weight = self.compute_weight(distance)
            # 计算权重

            distances.append((weight, self.y_model[i]))
            # 将权重和对应的标签作为元组添加到列表中

        distances.sort(key=lambda x: x[0], reverse=True)
        # 用lambda表达式让这个列表按照权重进行降序排序

        k_neighbors = distances[:3]
        # 选择排序后权重最大的三个邻居

        weighted_votes = {}
        # 创建一个空字典用于记录每个标签的加权投票结果

        for weight, label in k_neighbors:
        # 遍历三个邻居的权重和标签

            if label in weighted_votes:
                weighted_votes[label] += weight
                # 如果这个标签已经在字典中存在 累加这个权重即可

            else:
                weighted_votes[label] = weight
                # 反之如果不在的话就创建一个这个标签的字典来保存当前权重

        predictions = max(weighted_votes, key=weighted_votes.get)
        # 从加权投票结果中选出权重最大的标签作为最终的预测结果

        return predictions
        # 返回这个结果

def digit_test():
    warnings.filterwarnings("ignore")
    recognizer = DigitRecognizer('../mnist_784.pth')
    filename = '../handwritten_digits/digit_1.png'
    prediction = recognizer.predict_digit(filename)
    print(f'测试结果为: {prediction}')
    # 关于预测图像 之前的文件也详细说过了 就不再讲了

if __name__ == '__main__':  # 主函数
    digit_test()
