# 基于Qt可视化的常见OpenCV图像处理实现（附源码）

本项目为初学OpenCV时自己实现的一套小程序，里面包含多种可对图像操作的方式，主要以展示为主，实际应用可扩展性并不强（水平有限），适合面向初学OpenCV的小伙伴们，可以通过学习本项目的代码，对OpenCV一些简单的操作有更深刻的认识。

在完成本项目时，为了加深对OpenCV的了解，很多简单的函数并没有调库，而是通过对像素基础操作自行实现的，如有小bug大佬轻喷。

**<u>*项目已开源，链接见文末，文案转载需标明出处！*</u>**

下面是相关介绍和内容展示：

## 项目内容

主要实现编程一套基于OpenCV的图像处理和计算机视觉程序。实验内容包括以下几个部分：

（1）图像预处理：打开图片、图片灰度处理、显示灰度直方图、灰度均衡、梯度锐化、Laplace锐化等；

（2）边缘检测：其中包括Roberts算子、Sobel算子、Laplace算子、Prewitt算子、Canny算子、Krisch算子等；

（3）添加噪声：包括椒盐噪声、高斯噪声等；

（4）滤波处理：包括均值滤波、中值滤波、边窗滤波、形态学滤波、高斯滤波等

（5）摄像标定：实现摄像机标定。立体匹配等；

（6）图像变换：主要为仿射变换和透视变换；

（7）背景处理：实现阈值分割、OSTU和Kittler静态阈值分割、帧间差分和高斯混合背景等；

（8）特征明显操作：包括LBP、直方图检测、模板匹配、颜色匹配、Gabor滤波等；

（9）特征提取：实现SIFT算法、ORB算法、坐标点SVM、Haar算法等。

## 系统环境及运行要求

1、操作系统：MacOS 11.5.2

2、开发平台：Qt 5.15.1 (Clang 11.0 (Apple), 64 bit)

3、机器视觉库：Opencv-4.5.3

4、编程语言：C++

## 结果展示

1、界面设计

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsAP36hh.jpg)

2、灰度处理

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wps2Ghw6Y.jpg)

3、边缘检测（Sobel算子）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsY4qyFz.jpg)

4、添加噪声（椒盐噪声）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsZMyoev.jpg)

5、滤波处理（中值滤波）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpspI8TOQ.jpg)

6、滤波处理（形态学滤波）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsFTtxkC.jpg)

7、双目摄像机标定

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsOcmFzT.jpg)

8、图像变换（透视变换）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpstXDt1S.jpg)

9、阈值分割（OSTU）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsGFOoJC.jpg)

10、高斯混合背景

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpscem4YS.jpg)

11、模板匹配

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpszJQRf8.jpg)

12、隐身效果

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsCKgaTV.jpg)

13、颜色匹配

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsaJBtFz.jpg)

14、特征提取（SIFT）

![img](https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/opencv图像处理基础实现/wpsPEKJ1W.jpg)

其余效果还有很多，不再一一展示了。

## 附录

code：