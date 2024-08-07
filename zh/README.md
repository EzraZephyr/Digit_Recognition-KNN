# 本篇git适用于人工智能算法初学者

讨论的是如何使用K-最近邻算法实现简单手写数字识别

由于我们使用的数据集是 MNIST 数据集，并且采用了 KNN 算法进行手写数字识别。MNIST 数据集包含灰度图像，因此对色差非常敏感。如果手写数字图像的质量不高或者存在噪声，可能会导致识别错误。不过，对于初学者来说，这类数据集已经足够用于理解和学习 KNN 算法的应用。

## 以下为文件的简单介绍，所有文件都已标注较为详细的注释

- **Learning**：为训练模型的文件
- **handwritten_digits**：为用于预测图片数字的文件，其从MNIST数据集中随机抽取了30张灰度图，可以用这里的图片进行预测
- **Digit_Test**：为测试模型的文件
- **Digit_WeightedKNN**：使用加权 KNN 进行手写数字识别的文件。(如果对做成一个小软件没有兴趣的话，到这步已经可以了）
- **Digit_Recognition**：为最终效果的文件，使用GUI展示了一个窗口，可以实现文件的上传和结果的展示
