import pandas as pd    # 处理数据集
import joblib   # 加载和保存训练好的模型
from sklearn.datasets import fetch_openml   # 手写数字图像
from sklearn.model_selection import train_test_split    # 分割训练数据和测试数据并调整比例
from sklearn.neighbors import KNeighborsClassifier  # K-最近邻分类器 也就是KNN算法 根据最近的K个邻居进行预测
from sklearn.metrics import accuracy_score  # 计算模型的准确率

mnist = fetch_openml('mnist_784',version=1)
# 取出名为mnist_784的数据集 版本为1 该数据集里包含了7万张0~9的手写数字图像
# 一个有趣的背景 该数据集中的6万张训练图像是由美国人口普查局的工作人员所写 而一万张测试图像是来自美国高中生

X = pd.DataFrame(mnist['data'])
# 提取该数据集中的特征 这些特征为二维向量且在数据中的索引为data 所以需要用DataFrame来处理他们
y = pd.Series(mnist.target).astype('int')
# 对于此数据集中的标签 也就是一个28*28的二维向量所对应的目标值 是一维向量 所以用Series处理即可
# 因为目标值被设计为字符串类型 所以要用astype('int')将其转化为int类型

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# 使用train_test_split函数将特征和标签(也就是上面的X和y)以80%和20%的比例进行分割
# 使X_train和y_train用于训练 X_test和y_test用于测试

estimator = KNeighborsClassifier(n_neighbors=3)
# 实例化一个K-最近邻分类器 并将n_neighbors设置为3 这也就意味着在以后的数据测试或模型使用中
# 该算法会自动查找其在特征空间中距离最近的三个邻居(一般使用欧氏距离)
# 并通过这三个邻居的类别来决定新数据点的类别

estimator.fit(X_train,y_train)
# 开始正式训练模型 使用这个分类器的fit方法 传入训练数据和其对应的标签
# 让模型学习并记住数据特征和目标值之间的关系 以便以后可以用来做预测

y_pred = estimator.predict(X_test)
# 模型训练完成之后 就需要使用上面拆分出的测试数据来进行预测 计算模型的准确度。
# 通过调用分类器的predict方法 传入测试数据X_test 就会根据之前学到的规律 返回对每个测试样本的预测结果。

print(accuracy_score(y_test,y_pred))
# 通过测试结果的对比来计算出模型的精准度

joblib.dump(estimator, '../mnist_784.pth')
# 最后 调用joblib库中的dump方法 传入训练好的模型并设置文件名为‘mnist_784.pth'
# 这样模型就被保存在了磁盘上 以后就可以在不需要重新训练模型的情况下 直接加载使用这个已经训练好的模型





#%%
