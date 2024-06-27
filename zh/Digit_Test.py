import warnings   # 忽略警告信息
import joblib   # 加载和保存训练好的模型
import numpy as np  # 操作数组
from PIL import Image  # 处理图像文件

def digit_test():
    warnings.filterwarnings("ignore")
    # 因为训练时的特征名称和现在的特征名称不一样 会报warning但是不影响运行和结果
    # 所以直接忽略这个warning就可以了 如果想解决的话在测试模型前的代码中加上
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])] 即可

    model = joblib.load('../mnist_784.pth')
    # 使用joblib方法加载刚才训练好的模型

    filename = '../handwritten_digits/digit_1.png'
    # 将图片文件的路径储存在filename里 方便下面直接调用

    img = Image.open(filename).convert('L')
    # 用Image.open方法将该路径上的图片打开 并且使用convert方法将其转换为灰度图

    img = img.resize((28, 28))
    # 将图片大小压缩为28*28的格式 以符合模型的输入要求

    img = np.array(img).reshape(1,-1)
    # 用np.array方法把img转成数组类型 reshape方法中第一个‘1’ 是用于将这个二位数组拉伸成一个一维数组
    # 第二个‘-1’是让Numpy自动计算出剩余维度的大小 使得数组展平成一个包含784个元素的数组

    predict = model.predict(img)
    # 将处理好的img投入到模型中进行预测 然后该模型会根据设定中最近的三个‘邻居'
    # 进行投票后返回最终结果

    print(f'测试结果为: {predict[0]}')
    # 因为返回的是一个数组的形式 所以其中的第一个数输出 就是结果

if __name__ == '__main__': # 主函数
    digit_test()