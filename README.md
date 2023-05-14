# DeepLearing-Summer
大二下暑期学习计划

## 基于TensorFlow2学习基础神经网络模型

## [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)

观看YouTube上的教学视频（10个）, [YouTube - MIT Introduction to Deep Learning | 6.S191](https://www.youtube.com/watch?v=7sB052Pz0sQ&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)，[阿里云 - 1080P](https://www.aliyundrive.com/s/L5WQKeEKAjn).

掌握基本的深度学习模型及算法原理 ---- 三种基本神经网络（前馈型神经网络，卷积神经网络，循环神经网络）、强化学习。通过项目对模型的具体使用进行练习，完成TensorFlow中代码实现，提出自己的体会和可改进的地方.

```python
# 包版本：
mitdeeplearning: 0.2.0
tensorflow: 2.x
```



#### 项目实现目标 - 三个Software Lab

- ##### Music Generation（音乐生成）

- ##### Recognition Systems（人脸识别）

- ##### ~~Self-Driving Control（自动驾驶控制）~~（最后改为平衡滑块问题）

  对应 GitHub 上的代码进行学习.

#### 算法原理学习

《神经网络与深度学习》 ----  邱锡鹏

#### Tensorflow学习

- 《深度学习实战 - 基于 TensorFlow2 和 Keras》 ---- Gulli, Kapoor, Pal

- YouTube课程，推荐 [TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial](https://www.youtube.com/watch?v=tPYj3fFJGjk&t=8210s)，虽然第一部分的estimator官方已不推荐使用，但视频讲解的非常详细，本GitHub项目中 [TensorFlow](./TensorFlow/) 下有我自己对应做的中文笔记（非常详细），也有YouTube原版英文笔记.

YouTube课程视频（6小时52分）下载链接：[阿里云 - 720P](https://www.aliyundrive.com/s/SMAkuJeQGub)，[阿里云 - 1080P](https://www.aliyundrive.com/s/yVRLaMNyc7E)，[bilibili](https://www.bilibili.com/video/BV1EB4y1r7w7/?vd_source=92e1ce2ebcdd26a23668caedd3c9e57e).

- TensorFlow官网API[参考文档](https://tensorflow.google.cn/api_docs/python/tf)，使用方法：在左侧栏，找到自己要用的函数API，点进去查看详细使用说明.

- MIT课程源代码.

#### Pandas学习

学习YouTube上[Python Pandas Tutorial](https://www.youtube.com/watch?v=ZyhVh-qRZPA&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS)，一共11集. 需要掌握Pandas对训练数据的处理，以适用于模型训练之中.

## 学习成果展示

[pdf版 总结报告](https://github.com/wty-yy/DeepLearing-Summer/blob/master/Summer%20-%20DeepLearning.pdf)，三个Lab的项目在blog上文章分别为：

1. [基于循环神经网络RNN的简单音乐生成](https://wty-yy.github.io/posts/42812/).
2. [基于卷积神经网络CNN和去偏变分自动编码机DB-VAE的简单人脸识别模型](https://wty-yy.github.io/posts/52484/).
3. [强化学习 - Deep Q-Learning Network算法 解决平衡小推车问题（Cartpole）](https://wty-yy.github.io/posts/44956/).

生成的[乐曲文件](https://github.com/wty-yy/DeepLearing-Summer/tree/master/MIT%206S191/Lab1/songs).

VAE渐变效果图：

![渐变效果图](./README.figure/VAE.png)

小车平衡效果图：

![Mini_batch_perfect_cut](./README.figure/Mini_batch_perfect_cut.gif)

![Mini_batch18_best_cut](./README.figure/Mini_batch18_best_cut.gif)
