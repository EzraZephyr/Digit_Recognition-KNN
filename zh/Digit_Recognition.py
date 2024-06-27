import warnings  # 忽略警告信息
import sys  # 系统相关功能
import joblib  # 加载和保存训练好的模型
import numpy as np  # 操作数组
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel  # GUI构建模块 下面用到了就了解了
from PyQt5.QtGui import QPixmap  # 处理图像显示
from PIL import Image  # 处理图像文件

class MainWindow(QWidget):
    # 创建GUI主窗口的类 并且继承QWidget的方法 可以完全自定义窗口的外观和操作

    def __init__(self):
        super().__init__()
        # 在执行这个初始化函数的同时 调用父类(QWidget)方法

        self.init_ui()
        self.model = joblib.load('../mnist_784.pth')
        # 使用joblib方法加载训练好的模型

    def init_ui(self):
        self.setWindowTitle('手写数字识别')
        # 设置这个窗口的标题 也就是打开窗口后在最上方居中边框的位置

        self.resize(1000, 600)
        # 调整窗口的大小

        layout = QVBoxLayout()
        # 创建垂直布局 这样后面被添加到垂直布局里的子部件就会从上到下垂直排列
        # 并且根据窗口大小的变化实时调整子部件的大小和位置 并均匀分布 防止重叠

        self.btn = QPushButton('加载图片', self)
        # 创建一个按钮 并在按钮上水平居中显示"Add Photo"的文本
        # 对括号中的self做一个简单的解释: 这个self是指定了这个类作为按钮的"父控件“
        # 将按钮添加到这个窗口中 等这个窗口被关闭时 该按钮就被自动销毁 用于避免内存泄露

        self.btn.setFixedSize(200, 200)
        # 调整按钮的大小

        self.btn.clicked.connect(self.load_Image)
        # 将该按钮的点击信号连接到self.loadImage这个函数 这样当按钮被点击时就会触发这个函数

        layout.addWidget(self.btn)
        # 将这个按钮添加到布局中

        self.resultLabel = QLabel('测试结果为:', self)
        # 创建一个标签用于显示最后的结果

        layout.addWidget(self.resultLabel)
        # 将结果标签也添加到布局中

        self.imageLabel = QLabel(self)
        # 创建一个标签用于显示测试的图片

        layout.addWidget(self.imageLabel)
        # 将这个图片标签也添加到布局中

        self.setLayout(layout)
        # 将刚才创建的布局设置为当前窗口的布局管理器 这样刚才添加到布局设置里的标签就可以被自动调整后显示了
    def load_Image(self):
        options = QFileDialog.Options()
        # 这个方法就是创建点击图片的时候触发的那个文件选择框

        filename, _ = QFileDialog.getOpenFileName(self, "请选择图片", "", "All Files (*)", options=options)
        # 打开文件选择框 并且选取文件传递给对话框 ”“代表默认目录 ”All Files (*)"则可以显示选择所有类型的文件

        if filename:
            pixmap = QPixmap(filename)
            # 使用QPixmap方法加载选择的图像 用QPixmap的主要原因是因为其可以和QLabel兼容 可以直接加载到imageLabel中

            self.imageLabel.setPixmap(pixmap)
            # 将加载的图像设置为imageLabel 这样可以在窗口中显示出来

            self.imageLabel.adjustSize()
            # 将ImageLabel调整为合适的大小以适应图像

            prediction = self.predict_Digit(filename)
            # 调用predictDigit函数进行预测 并将值返回给prediction

            self.resultLabel.setText(f'测试结果为:{prediction}')
            # 将result_Label的文本显示内容添加上刚刚预测的结果


    def predict_Digit(self, filename):
        img = Image.open(filename).convert('L')
        # 用Image.open方法将该路径上的图片打开 并且使用convert方法将其转换为灰度图

        img = img.resize((28, 28))
        # 将图片大小压缩为28*28的格式 以符合模型的输入要求

        img = np.array(img).reshape(1, -1)
        # 用np.array方法把img转成数组类型 reshape方法中第一个‘1’ 是用于将这个二位数组拉伸成一个一维数组
        # 第二个‘-1’是让Numpy自动计算出剩余维度的大小 使得数组展平成一个包含784个元素的数组

        prediction = self.model.predict(img)
        # 将处理好的img投入到模型中进行预测 然后该模型会根据设定中最近的三个‘邻居'
        # 进行投票后返回最终结果

        return prediction[0]
        # 因为返回的是一个数组的形式 所以其中的第一个数输出 就是结果 返回该结果即可

if __name__ == '__main__':  # 主函数
    warnings.filterwarnings("ignore")
    # 因为训练时的特征名称和现在的特征名称不一样 会报warning但是不影响运行和结果
    # 所以直接忽略这个warning就可以了 如果想解决的话在测试模型前的代码中加上
    # X.columns = [f'pixel{i}' for i in range(X.shape[1])] 即可

    app = QApplication(sys.argv)
    # 创建一个QApplication对象 负责管理该程序的控制流和其他设置

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
    # app.exec_()是进入该程序的主循环 开始上述事件的处理 exit时确保该程序可以干净地退出