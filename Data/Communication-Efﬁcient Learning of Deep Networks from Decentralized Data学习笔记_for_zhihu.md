# Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记

[google原文](http://proceedings.mlr.press/v54/mcmahan17a.html)

[toc]

## 联邦学习基础学习（预备知识）

当下，大部分AI是靠数据来喂的，而且是大量优质数据。现实生活中，除了少数巨头公司能够满足，绝大多数企业都存在**<u>数据量少，数据质量差</u>**的问题，不足以支撑人工智能技术的实现。

### 1、联邦学习概念

- 定义：联邦学习本质上是一种**分布式**机器学习技术，或机器学习**框架**。

- 目标：联邦学习的目标是在保证数据隐私安全及合法合规的基础上，实现共同建模，提升AI模型的效果。

### 2、联邦学习分类

我们把每个参与共同建模的企业称为参与方，根据多参与方之间数据分布的不同，把联邦学习分为三类：**横向联邦学习、纵向联邦学习和联邦迁移学习**。

- 横向联邦学习

横向联邦学习的本质是**样本的联合**，适用于参与者间业态相同但触达客户不同，即特征重叠多，用户重叠少时的场景，比如不同地区的银行间，他们的业务相似（特征相似），但用户不同（样本不同）

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129090254674.png" alt="image-20211129090254674" style="zoom:40%;" />

step1：参与方各自从服务器A下载最新模型；

step2：每个参与方利用本地数据训练模型，加密梯度上传给服务器A，服务器A聚合各用户的梯度更新模型参数；

step3：服务器A返回更新后的模型给各参与方；

step4：各参与方更新各自模型。

在传统的机器学习建模中，通常是把模型训练需要的数据集合到一个数据中心然后再训练模型，之后预测。在横向联邦学习中，可以看作是**基于样本的分布式模型训练**，分发全部数据到不同的机器，每台机器从服务器下载模型，然后利用本地数据训练模型，之后返回给服务器需要更新的参数；服务器聚合各机器上的返回的参数，更新模型，再把最新的模型反馈到每台机器。

在这个过程中，每台机器下都是**相同且完整的模型**，且机器之间不交流不依赖，在预测时每台机器也可以**独立预测**，可以把这个过程看作成基于样本的分布式模型训练。

- 纵向联邦学习

纵向联邦学习的本质是**特征的联合**，适用于用户重叠多，特征重叠少的场景，比如同一地区的商超和银行，他们触达的用户都为该地区的居民（样本相同），但业务不同（特征不同）。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129091459642.jpg" alt="image-20211129091459642" style="zoom:40%;" />

第一步：加密样本对齐。是在系统级做这件事，因此在企业感知层面不会暴露非交叉用户。

第二步：对齐样本进行模型加密训练：

step1：由第三方C向A和B发送公钥，用来加密需要传输的数据；

step2：A和B分别计算和自己相关的特征中间结果，并加密交互，用来求得各自梯度和损失；

step3：A和B分别计算各自加密后的梯度并添加掩码发送给C，同时B计算加密后的损失发送给C；

step4：C解密梯度和损失后回传给A和B，A、B去除掩码并更新模型。

**在整个过程中参与方都不知道另一方的数据和特征，且训练结束后参与方只得到自己侧的模型参数，即半模型。**故预测时需要双方协助完成：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129092222848.jpg" alt="image-20211129092222848" style="zoom:40%;" />

- 联邦迁移学习

当参与者间**特征和样本重叠都很少**时可以考虑使用联邦迁移学习，如不同地区的银行和商超间的联合。主要适用于以深度神经网络为基模型的场景。

迁移学习的核心是，找到源领域和目标领域之间的相似性。比如，我们如果已经会打乒乓球，就可以类比着学习打网球。再比如，我们如果已经会下中国象棋，就可以类比着下国际象棋。

---

## SGD相关知识（预备知识）

线性回归的目的是通过几个已知数据来预测另一个数值型数据的目标值。假设其满足公式：

<img src="https://www.zhihu.com/equation?tex=h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}
" alt="h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 就是权重，经过变换可以写为：

<img src="https://www.zhihu.com/equation?tex=h(x)=\sum_{i=0}^{n} \theta_{i} x_{i}=\theta^{T} x
" alt="h(x)=\sum_{i=0}^{n} \theta_{i} x_{i}=\theta^{T} x
" class="ee_img tr_noresize" eeimg="1">
相当于求解 <img src="https://www.zhihu.com/equation?tex=h" alt="h" class="ee_img tr_noresize" eeimg="1"> 从而预测，即让 <img src="https://www.zhihu.com/equation?tex=h(x)" alt="h(x)" class="ee_img tr_noresize" eeimg="1"> 与 <img src="https://www.zhihu.com/equation?tex=y(x)" alt="y(x)" class="ee_img tr_noresize" eeimg="1"> 去逼近。我们定义了一个函数来描述这个差距，这个函数称为损失函数，表达式如下：

<img src="https://www.zhihu.com/equation?tex=J(\theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
" alt="J(\theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
" class="ee_img tr_noresize" eeimg="1">
我们要求解使得 <img src="https://www.zhihu.com/equation?tex=J(θ)" alt="J(θ)" class="ee_img tr_noresize" eeimg="1"> 最小的 <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> 值，梯度下降算法大概的思路是：我们首先随便给 <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> 一个初始化的值，然后改变 <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> 值让 <img src="https://www.zhihu.com/equation?tex=J(θ)" alt="J(θ)" class="ee_img tr_noresize" eeimg="1"> 的取值变小，不断重复改变 <img src="https://www.zhihu.com/equation?tex=θ" alt="θ" class="ee_img tr_noresize" eeimg="1"> 使 <img src="https://www.zhihu.com/equation?tex=J(θ)" alt="J(θ)" class="ee_img tr_noresize" eeimg="1"> 变小的过程直至 <img src="https://www.zhihu.com/equation?tex=J(θ)" alt="J(θ)" class="ee_img tr_noresize" eeimg="1"> 约等于最小值。更新公式为：

<img src="https://www.zhihu.com/equation?tex=\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)
" alt="\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)
" class="ee_img tr_noresize" eeimg="1">
公式中的 <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> 为步长，最终将 <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 带入得：

<img src="https://www.zhihu.com/equation?tex=\theta_{j}:=\theta_{j}+\alpha\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
" alt="\theta_{j}:=\theta_{j}+\alpha\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
" class="ee_img tr_noresize" eeimg="1">
上述表达式只针对样本数量只有一个的时候适用，那么当有m个样本值时该如何计算预测函数？共有两种方法：

- **BGD**

计算没一个 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> ，迭代更新。此方式对于权重个数多的情况不适用。

- **SGD**

在计算下降最快的方向时时随机选一个数据进行计算，而不是扫描全部训练数据集，这样就加快了迭代速度。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129164958246.png" alt="image-20211129164958246" style="zoom:40%;" />



## Abstract摘要

丰富的数据通常对隐私敏感，且数量庞大。我们提倡另一种方法，将训练数据**分布在移动设备**上，并通过聚集本地计算的更新来学习共享模型。称其为联邦学习。

我们提出了一种基于迭代模型平均的深度网络联合学习的实用方法，效果显著。

## 1 Introduction

- **一、研究内容**

我们研究了一种学习技术，该技术允许用户集体从这些丰富的数据中获得培训的共享模型的好处，而不需要集中存储它。我们将我们的方法称为联邦学习。

学习任务是由参与设备(我们称为<u>*客户机clients*</u>)的松散联合解决的，这些设备由<u>*中央服务器server*</u>协调。**每个客户端都有一个本地的培训数据集，它永远不会上传到服务器**。

这种方法的主要优点是将模型训练与直接访问原始训练数据的需要分离开来。

主要贡献：

1. 将移动设备分散数据的培训问题确定为一个重要的研究方向
2. 选择一种简单实用的算法应用于此设置
3. 对所提出的方法进行了广泛的实证评价

更具体地说，我们引入了***<u>联邦平均化算法</u>***，该算法将每个客户端上的局部随机梯度下降(SGD)与执行模型平均化的服务器相结合。**该算法对不平衡和非iid数据分布具有鲁棒性。**并且可以将训练深度网络处理分散数据所需的通信次数减少几个数量级。

> **什么是IID（独立同分布）？**
>
> 输入空间X的所有样本服从一个隐含未知的分布，训练数据所有样本都是独立地从这个分布上采样而得。即是指一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立。在传统有监督机器学习研究里，IID是一个重要假设，因为人们希望训练集和测试集满足IID。
>
> 对于none-IID，即全部随机，数据集选取（label比例）也随机，测试机训练集划分也随机。

---



- **二、联邦学习**

联邦学习的数据应有以下标准：

1. 对来自移动设备的真实数据进行培训，比对通常在数据中心可用的代理数据进行培训具有明显的优势。
2. 最好不要纯粹为了模型培训的目的而将其记录到数据中心。
3. 对于监督任务，数据上的标签可以从用户交互中自然地推断出来。

---



- **三、隐私**

联邦学习具有明显的隐私优势，故永远不会包含比原始训练数据更多的信息。而且聚合算法不**需要更新的源**，因此可以在Tor[7]等混合网络上或通过可信的第三方传输更新，而无需识别元数据。

---



- **四、联合优化**

我们将联邦学习中隐含的优化问题称为联邦优化，将其与分布式优化联系起来(并进行对比)。其具备几个属性：

1. **non-IID**。数据分布不平均
2. **不平均**。一些用户会比其他人更频繁地使用这项服务或应用程序，从而导致不同数量的本地培训数据。
3. 大规模分布式。期望参与优化的客户端数量比每个客户端的平均示例数量多得多
4. 通讯限制（网络）。

**优化步骤**（按照轮数进行迭代）：

1. 固定总数 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 个客户端，每个客户端都有本地数据集。
2. 每次选取分数 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> （比例）个客户端
3. 服务器将当前的全局算法发送给每个客户端。
4. 每个被选定的客户端执行本地计算，并将服务器更新。 

> 经过实验，发现每次只选择一部分客户端效果更好

对于凸神经网络，有如下目标：

<img src="https://www.zhihu.com/equation?tex=minf(w)\quad f(w)=\frac{1}{n}\sum_{i=1}^n f_i(w)
" alt="minf(w)\quad f(w)=\frac{1}{n}\sum_{i=1}^n f_i(w)
" class="ee_img tr_noresize" eeimg="1">
也就是说对于一般的机器学习 <img src="https://www.zhihu.com/equation?tex=f_i(w)=l(x_i,y_i,w)" alt="f_i(w)=l(x_i,y_i,w)" class="ee_img tr_noresize" eeimg="1"> ，其损失函数 <img src="https://www.zhihu.com/equation?tex=(x_i,y_i)" alt="(x_i,y_i)" class="ee_img tr_noresize" eeimg="1"> 由 <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 决定。假设数据分布在 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 个客户端， <img src="https://www.zhihu.com/equation?tex=P_k" alt="P_k" class="ee_img tr_noresize" eeimg="1"> 代表客户端 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 数据点的集合， <img src="https://www.zhihu.com/equation?tex=n_k" alt="n_k" class="ee_img tr_noresize" eeimg="1"> 表示客户端 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 的数据个数集合。则上式可以写为：

<img src="https://www.zhihu.com/equation?tex=f(w)=\sum_{k=1}^{K} \frac{n_{k}}{n} F_{k}(w)\quad F_{k}(w)=\frac{1}{n_{k}} \sum_{i \in \mathcal{P}_{k}} f_{i}(w)
" alt="f(w)=\sum_{k=1}^{K} \frac{n_{k}}{n} F_{k}(w)\quad F_{k}(w)=\frac{1}{n_{k}} \sum_{i \in \mathcal{P}_{k}} f_{i}(w)
" class="ee_img tr_noresize" eeimg="1">
如果划分 <img src="https://www.zhihu.com/equation?tex=P_k" alt="P_k" class="ee_img tr_noresize" eeimg="1"> 是所有用户数据的随机取样，则目标函数 <img src="https://www.zhihu.com/equation?tex=f(w)" alt="f(w)" class="ee_img tr_noresize" eeimg="1"> 就等价于损失函数关于 <img src="https://www.zhihu.com/equation?tex=P_k" alt="P_k" class="ee_img tr_noresize" eeimg="1"> 的期望：

<img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\mathcal{P}_{k}}\left[F_{k}(w)\right]=f(w)
" alt="\mathbb{E}_{\mathcal{P}_{k}}\left[F_{k}(w)\right]=f(w)
" class="ee_img tr_noresize" eeimg="1">
这就是传统的分布式优化问题的***独立同分布假设***，对于Non-IID情况可以是 <img src="https://www.zhihu.com/equation?tex=F_k" alt="F_k" class="ee_img tr_noresize" eeimg="1"> 是 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 的任意坏的近似。

---



- **五、通信成本**

一般数据中心中，通讯占少数，计算占大头。但是在联邦优化中，通讯占主导地位：

1. 我们通常会受到1 MB / s或更小的**上传带宽的限制**；
2. 客户通常只会在**充电，插入电源和不计量的Wi-Fi连接时**自愿参与优化；
3. **希望每个客户每天只参加少量的更新回合**。

所以，**<u>*我们希望减少通讯次数*</u>**，具体做法是：

1. 增加并行度，每一个轮中使用更多的客户端；
2. 增加每个用户的计算量。

---



- **六、研究成果**

众多（参数化）算法中，我们最终考虑的一个是简单一次平均**simple one-shot averaging**，其中每个客户解决的模型**<u>*将其本地数据的损失降到最低*</u>**（可能是正则化的），然后**<u>*将这些模型取平均值以生成最终的全局模型*</u>**。



## 2 The FederatedAveraging Algorithm

SGD可以很好地应用在联邦优化中，由于在每轮通讯中都会进行单个的梯度计算（在随机的客户端上），这种方法在计算上是有效的，但需要非常大量的训练才能产生好的模型。

---



- **federatedSGD算法（FedSGD）**

利用基线算法（baseline）进行大批量同步SGD。我们在每轮中选择C个客户端，并计算这些客户端持有的所有数据的损失梯度。因此，C控制全局批次大小，C=1对应于全批次(非随机)梯度下降。

---



- **FedSGD和FedAvg的区别**

对于FedSGD来说，其算法原理为：

1. 计算每一个client的梯度值：


<img src="https://www.zhihu.com/equation?tex=g(k)=\nabla F_{k}\left(w_{t}\right)
" alt="g(k)=\nabla F_{k}\left(w_{t}\right)
" class="ee_img tr_noresize" eeimg="1">

2. 传给server，进行聚合：


<img src="https://www.zhihu.com/equation?tex=w_{t+1} \leftarrow w_{t}-\eta \sum_{k=1}^{K} \frac{n_{k}}{n} g_{k}
" alt="w_{t+1} \leftarrow w_{t}-\eta \sum_{k=1}^{K} \frac{n_{k}}{n} g_{k}
" class="ee_img tr_noresize" eeimg="1">

对于FedAvg来说，其算法原理为：

1. 用户client在本地先聚合多次：


<img src="https://www.zhihu.com/equation?tex=w_{t+1}^k \leftarrow w_{t}-\eta g_{k}
" alt="w_{t+1}^k \leftarrow w_{t}-\eta g_{k}
" class="ee_img tr_noresize" eeimg="1">

2. 传递给server进行最终聚合：


<img src="https://www.zhihu.com/equation?tex=w_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_{k}}{n} w_{t+1}^k
" alt="w_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_{k}}{n} w_{t+1}^k
" class="ee_img tr_noresize" eeimg="1">

**主要区别：FedAvg相当于FedSGD在用户本地多次梯度更新。**

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129195943177.png" alt="image-20211129195943177" style="zoom:50%;" />

计算量由三个关键参数控制：C，在每轮上执行计算的客户端的比例；E，每个客户端在每轮上对其本地数据集进行的训练通过数；B，客户端更新所使用的本地小批量大小。我们写入B=∞以指示将整个本地数据集作为单个小批处理，取B=∞和E=1，它正好对应于FedSGD。

**B=∞：** 代表minibatch=用户本地全部数据

**B=∞ & E = 1：** FedAvg 等价于 FedSGD

---



- **模型效果分析**

对于一般的非凸目标函数，**参数空间中的平均模型可能会产生任意不好的模型结果**。 当我们平均两个**从不同初始条件训练**的MNIST数字识别模型时，我们恰好看到了这种不良结果。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129194701527.png" alt="image-20211129194701527" style="zoom:50%;" />

通过平均两个模型 <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=w^\prime" alt="w^\prime" class="ee_img tr_noresize" eeimg="1"> 的参数而产生的模型的全MNIST训练集上的损失，结合的方式为： <img src="https://www.zhihu.com/equation?tex=\theta w+(1-\theta) w^{\prime}" alt="\theta w+(1-\theta) w^{\prime}" class="ee_img tr_noresize" eeimg="1"> 。对于左边的图， <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=w^\prime" alt="w^\prime" class="ee_img tr_noresize" eeimg="1"> 的使用不同的随机种子初始化；对于右边的图，使用了共享种子。在共享初始化的情况下，对模型求平均值会显著减少整个训练集的损失(比任何一个父模型的损失都好得多)。

**<u>*FedAvg伪代码：*</u>**

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211129201045552.png" alt="image-20211129201045552" style="zoom:40%;" />

> 1. server要做的：
>
> - 初始化 <img src="https://www.zhihu.com/equation?tex=w_0" alt="w_0" class="ee_img tr_noresize" eeimg="1"> 
> - 选取 <img src="https://www.zhihu.com/equation?tex=C\times K" alt="C\times K" class="ee_img tr_noresize" eeimg="1"> 个client。
> - 每个client**并行更新**。
> - 按比例将所有的 <img src="https://www.zhihu.com/equation?tex=w_{t+1}^k" alt="w_{t+1}^k" class="ee_img tr_noresize" eeimg="1"> 相加
>
> 2. client更新要做的：
>
> - 将第 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 个client下的用户数据 <img src="https://www.zhihu.com/equation?tex=\mathcal{P}_{k}" alt="\mathcal{P}_{k}" class="ee_img tr_noresize" eeimg="1"> 划分为 <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> 
> - 从1到 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 次epoch对模型梯度更新

---



## 3 Experimental Results

实验数据：

1. 首先选择一个大小适中的代理数据集，以便我们可以彻底研究FedAvg算法的超参数。 （虽然每次训练的规模都比较小，但我们为这些实验训练了2000多个个体模型。 ）
2. 然后，我们介绍基准CIFAR-10图像分类任务的结果。
3. 为了证明FedAvg在数据自然分区的实际问题上的有效性，评估了一项**大型语言建模任务**。

---



- **利用MNIST图像分类**

研究包括两个数据集上的三个模型族。前两个用于MNIST数字识别任务：

1. 一个简单的多层感知器，有两个隐藏层，每个层有200个单元，使用ReLu激活，称为2NN。
2. 有两个5x5卷积层CNN

我们还需要指定数据如何在客户端client上分布。我们研究了在客户端上划分MNIST数据的两种方法：

1. IID，数据被混洗，然后划分为100个客户端，每个客户端接收600个示例；
2. 非IID，按数字标签对数据进行排序，将其划分为200个大小为300的碎片，并为100个客户端中的每个分配2个碎片（每个人手里最终只有两种label）

- **语言分类**

威廉·莎士比亚全集构造数据集，每部剧中的每个演讲角色构建了一个客户数据集。

一共1146个client（剧中角色），对于每个客户，我们将数据分成一组训练行（角色的前80%行）和测试行（最后20%），许多角色只有几行，少数角色有大量的行，故是Non-IID的；同理，使用相同的训练/测试分割，我们还形成了数据集的平衡和IID版本，也有1146个客户端。

---



**<u>*实验结果：*</u>**

- 分析参数 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 的影响，固定参数 <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211130143727881.png" alt="image-20211130143727881" style="zoom:40%;" />

 <img src="https://www.zhihu.com/equation?tex=C=0.0" alt="C=0.0" class="ee_img tr_noresize" eeimg="1"> 对应每轮一个客户端，我们对MNIST数据使用100个客户端，因此这些行对应于1、10、20、50和100个客户端。每个表条目给出了2NN达到97%和CNN达到99%的测试集准确度所需的**通信回合数**，以及相对于 <img src="https://www.zhihu.com/equation?tex=C=0.0" alt="C=0.0" class="ee_img tr_noresize" eeimg="1"> 基线的加速。

同时，使用较小的批量 <img src="https://www.zhihu.com/equation?tex=B = 10" alt="B = 10" class="ee_img tr_noresize" eeimg="1"> 比 <img src="https://www.zhihu.com/equation?tex=B = ∞" alt="B = ∞" class="ee_img tr_noresize" eeimg="1"> 时效果要好。（ <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> 是每个客户端训练所用数据量）

- 固定参数 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> ，分析参数 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> 。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211130150651673.png" alt="image-20211130150651673" style="zoom:60%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211130151117014.png" alt="image-20211130151117014" style="zoom:50%;" />

图中说明每轮添加更多本地SGD更新可以大大降低通信成本，表2量化了这些加速。其中每个客户端每回合的预期更新次数 <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1"> 可以表示为：

<img src="https://www.zhihu.com/equation?tex=u=E n /(K B)
" alt="u=E n /(K B)
" class="ee_img tr_noresize" eeimg="1">
**<u>*小结*</u>**

1. 通过MNIST图像结果发现，对于IID和non-IID的数据，提高E和B都能减少通信轮数。
2. 莎士比亚的**不平衡和非IID分布数据**（按角色扮演）更能代表我们期望用于实际应用中的数据分布。**在非IID和不平衡数据上的训练更加容易**。
3. **FedAvg收敛到比基准FedSGD模型更高的测试集准确性水平。**。

---



- **CIFAR-10 dataset** 进一步验证FedAvg的效果

**数据集：** 由具有三个RGB通道的10类32x32图像组成；50,000个训练样本和10,000个测试样本。将数据划分为100个clients，每个client包含500个训练样本和100个测试样本。

**模型：** 模型架构取自TensorFlow教程，包括2个卷积层、2个全连接层和1个线性转换层生成logit（共约106个参数）。

表3给出了Baseline SGD，FedSGD和FedAvg达到三项不同精度目标所需轮数，图4给出了FedAvg与FedSGD的学习率曲线。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211130155155959.png" alt="image-20211130155155959" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/Communication-Efﬁcient Learning of Deep Networks from Decentralized Data学习笔记_for_zhihu/image-20211130155210818.png" alt="image-20211130155210818" style="zoom:40%;" />

---



## 4 Conclusions and Future Work

- 各种模型架构（多层感知器，两种不同的卷积神经网络， 两层字符LSTM和大规模词级LSTM）的实验结果表明：当FedAvg使用**相对较少的交流轮次来训练高质量的模型时，联邦学习是实际可行的**。
- 尽管联合学习提供了许多实用的隐私优势，但通过**差分隐私、多方安全计算或者它们的组合**提供更强的隐私保证是未来工作的有趣方向。
