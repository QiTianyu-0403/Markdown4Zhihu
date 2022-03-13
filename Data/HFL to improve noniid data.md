# Federated learning with hierarchical clustering of local updates to improve training on non-IID data

对于传统FL来说non-iid情况很难处理，本位引入了层次聚类的步骤（FL+HC），通过客户端局部更新与全局联合模型的相似度来分离客户端集群，一旦分离，这些集群就会在专门的模型上进行独立和并行的训练。我们展示了本方案与没有聚类的FL相比，可以在更少的轮数下收敛。

## 层次聚类Hierarchical Clustering

<img src="http://bluewhale.cc/wp-content/uploads/2016/04/hierarchicalcluster.png" alt="hierarchicalcluster" style="zoom:60%;" />

一开始，每个样本都属于一个簇，通过计算欧氏距离、曼哈顿距离等方式，将距离最相近的作为一个簇。

如果簇之间有很多样本，选取最近的两个样本（或最远，或取中心店均值等等）作为距离。

当距离都小于某个阈值的时候，则结束聚类。

好处：与K-means相比不用设置聚类数目。

## Introduction

FL当下常见思路：

1、把中心端的模型下载下来，本地进行训练迭代：
$$
w_{t+1}^{k} \leftarrow w_{t}-\eta \nabla F_{k}\left(w_{t}\right)
$$
2、中心端接收所有客户端的模型，进行聚合学习：
$$
w_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_{k}}{n} w_{t+1}^{k}
$$

### Contribution

1、通过在一组通信轮之后，根据客户端对全局联合模型的更新，通过相似度对客户端进行聚类来实现减少收敛的轮次数。

2、全面描述在IID和各种非IID设置下的FL训练期间，层次聚类如何影响测试集准确性。

3、对FL的不同超参数和分层聚类算法对客户子集形成良好的专业化模型的影响进行了实证分析。

4、推荐好的默认超参数用于我们的方法的训练。

### 客户端统计的异质性

关键影响模型训练的因素就是noniid数据，所以本地模型和联合模型不相同：
$$
\mathbb{E}_{D_{k}}\left[F_{k}(w)\right] \neq f(w)
$$
一共有四种不同的noniid情况：

1、Feature distribution skew：$\mathcal{P}_i(x)$不同。

2、Label distribution skew：$\mathcal{P}_i(y)$不同。

3、Concept shift (same features, different label)：$\mathcal{P}_i(y|x)$不同。

4、Concept shift (same label, different features)：$\mathcal{P}_i(x|y)$不同。

<img src="https://img-blog.csdnimg.cn/bba369fb46ee443c8dff5bebe3ba12cc.png" alt="在这里插入图片描述" style="zoom:50%;" />

所以学习的过程中最好的方式是具有相同类似的数据分布的客户组的多个模型首先进行聚合。

### 分层聚类

通过对**从客户端收到的模型更新进行聚类**，可以实现对相关客户端子集的多模型训练。但当下并不知道需要聚类的数量，因此必须使用能够独立确定聚类数量的聚类算法。然而，一些自动确定聚类数量的聚类方法不能将离群样本分配到一个聚类中，而只是简单地将其标记为噪声。所以**层次聚类**是一种很好的选择。

在聚类的每一步，计算所有聚类之间的成对距离来判断它们的相似性。合并两个最相似的集群。这个过程总共持续N−1个步骤，直到剩下一个包含所有样本的集群。因此，构建了客户端之间的相似度层次结构

该方法的2个超参数很重要，一个是距离测量，一个是确定两个簇有多相似的链接机制。

## FL with HC

在FL设置下，假设**所有客户的目标接近全局目标**。然而，在存在非iid数据时，情况就不是这样了。所以引入了FL+HC的算法。

在通信轮第$n$轮处，引入聚类，使用所有客户端更新的局部模型来判断客户端之间的相似性，并使用分层聚类算法迭代合并最相似的客户端聚类到给定的距离阈值$T$。

一旦合并停止，集群$C$将会被独立开始训练，如果集群选取的好，在同一集群下所有客户端都应学习逼近同一模型：
$$
\forall c \in C, \mathbb{E}_{D_{k}}\left[F_{k}(w)\right]=f_{c}(w) \text { where } k \in c
$$
聚类时间复杂度为$O(n^3)$，一般该操作在服务器上，可以处理许多客户端的更新，而且只操作一次，对整体影响不大。

<img src="HFL to improve noniid data/image-20220309160005631.png" alt="image-20220309160005631" style="zoom:50%;" />

<img src="HFL to improve noniid data/image-20220309160023573.png" alt="image-20220309160023573" style="zoom:50%;" />

### 实验设置

数据集：MNIST

noniid设置：1、label的noniid，每个设备收集2种不同标签数据集。2、每个集群交换2种标签，例如第1集群交换数字3和9，这样保证了数据中存在$\mathcal{P}(y|x)$不均衡的情况。3、选用leaf下的FEMNIST数据集，包含62种类别（10+26+26），这样保证了$\mathcal{P}(x)$和$\mathcal{P}(y)$都不相同，更符合现实情况。

在上面描述的所有分区方案中，每个客户机都可以访问的测试数据集总是来自与训练数据相同的分布。

模型：CNN         

方式：先为$n$个通信轮次训练一个全局模型，然后各个客户端对全局模型进行一个3个阶段的训练，产生$\Delta w$，作为特征输入到层次聚类算法中，聚类算法返回多个聚类，每个聚类包含彼此最相似的客户端的子集。FL然后为每个集群独立进行总计50个通信轮。

## 结果

验证了训练50轮次之后的客户平均精度，还有50轮次后精度达到99%的客户数量。

### 聚类前不同客户分数和回合数的影响

聚类在此用欧几里得进行计算，阈值为3.0（FEMNIST为10.0），分数$\alpha$为0.1,0.2,0.5和1.0。聚类前的轮次数为1,3,5,10。

对于label-noniid的情况如下表和图：

<img src="HFL to improve noniid data/image-20220309165610444.png" alt="image-20220309165610444" style="zoom:50%;" />

<img src="HFL to improve noniid data/image-20220309165718039.png" alt="image-20220309165718039" style="zoom:50%;" />

<img src="HFL to improve noniid data/image-20220309165731007.png" alt="image-20220309165731007" style="zoom:50%;" />

可以发现FL+HC的方法更优，而且$n=1$的情况更好。

对于标签交换的noniid情况，因为预先一共将100个客户端分为了4组，每组内部进行一些标签交换，所以在此也希望能够通过FL+HC的方式找到四类。

<img src="HFL to improve noniid data/image-20220309194155268.png" alt="image-20220309194155268" style="zoom:50%;" />

<img src="HFL to improve noniid data/image-20220309194213821.png" alt="image-20220309194213821" style="zoom:50%;" />

FL算法一般在50轮次后达到99%的数量为0，但FL+HC能够训练高达80%的客户，达到99%的测试集准确性。

对于FEMNIST来说，其为更困难的方式，因为数量上也存在noniid。

<img src="HFL to improve noniid data/image-20220309194914015.png" alt="image-20220309194914015" style="zoom:50%;" />

<img src="HFL to improve noniid data/image-20220309194924440.png" alt="image-20220309194924440" style="zoom:50%;" />

上述实验表明，改变客户端比例$\alpha$似乎对最终结果影响不大，但在聚类步骤之前增加轮数可以比FL获得更大的增益（第2个实验很明显）。

### 不同层次聚类超参数的影响

下面实验将通信轮数设为10，将每轮参与的比例设为0.2。

<img src="HFL to improve noniid data/image-20220309200430784.png" alt="image-20220309200430784" style="zoom:50%;" />

对label—noniid进行了测试，上表表示，曼哈顿距离度量整体表现最好，其次是欧几里得距离，最后是余弦距离。同样，在曼哈顿距离度量下，达到目标精度的客户端数量最多。

<img src="HFL to improve noniid data/image-20220309201929949.png" alt="image-20220309201929949" style="zoom:50%;" />

对于交换noniid来说，上表表示，余弦距离更优。

<img src="HFL to improve noniid data/image-20220309202638124.png" alt="image-20220309202638124" style="zoom:50%;" />

对FEMNIST进行测试，上表所示，发现FL+HC相较于FL来说改善微乎其微，欧几里得距离使其有更好的准确率，曼哈顿距离（如下图）使其有更多的到达相应准确率的客户端数量。

<img src="HFL to improve noniid data/image-20220309202706672.png" alt="image-20220309202706672" style="zoom:50%;" />

综上，一个很好的超参数默认值是曼哈顿距离，$n=10$，$\alpha=0.2$。