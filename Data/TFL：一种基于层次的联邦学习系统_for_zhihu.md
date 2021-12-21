# TiFL：一种基于层次的联邦学习系统学习笔记

[toc]

## 预备知识汇总

### 1.分布式的同步和异步

深度学习模型最常采用的分布式**训练策略是数据并行**，因为训练费时的一个重要原因是训练数据量很大。数据并行就是在**很多设备上放置相同的模型**，并且各个设备采用不同的训练样本对模型训练。训练深度学习模型常采用的是batch SGD方法，采用数据并行，可以每个设备都训练不同的batch，然后收集这些梯度用于模型参数更新。

数据并行可以是同步也可以是异步的：

1. 同步指的是所有的设备都是采用相同的模型参数来训练，等待所有设备的mini-batch训练完成后，收集它们的梯度然后取均值，然后执行模型的一次参数更新。这相当于通过聚合很多设备上的mini-batch形成一个很大的batch来训练模型。但是实际上需要各个设备的计算能力要均衡，而且要求集群的通信也要均衡，类似于木桶效应，一个拖油瓶会严重拖慢训练进度。
2. 异步训练中，各个设备完成一个mini-batch训练之后，不需要等待其它节点，直接去更新模型的参数，这样总体会训练速度会快很多。但是异步训练的一个很严重的问题是**梯度失效**问题。刚开始所有设备采用相同的参数来训练，但是异步情况下，某个设备完成一步训练后，可能发现模型参数其实已经被其它设备更新过了，此时这个**梯度就过期**了，因为现在的模型参数和训练前采用的参数是不一样的。由于梯度失效问题，异步训练虽然速度快，但是可能陷入次优解。

### 2.差分隐私相关

列举张三单身调查的例子，为了避免这种情况出现，需要计入噪声，但还要使两种分布尽可能接近，故差分隐私的定义为：

<img src="https://www.zhihu.com/equation?tex=\operatorname{Pr}[\mathcal{M}(x) \in S] \leq e^{\epsilon} \operatorname{Pr}\left[\mathcal{M}\left(x^{\prime}\right) \in S\right]+\delta
" alt="\operatorname{Pr}[\mathcal{M}(x) \in S] \leq e^{\epsilon} \operatorname{Pr}\left[\mathcal{M}\left(x^{\prime}\right) \in S\right]+\delta
" class="ee_img tr_noresize" eeimg="1">
其中不看 <img src="https://www.zhihu.com/equation?tex=\delta" alt="\delta" class="ee_img tr_noresize" eeimg="1"> 的话，可以写为以下形式，即为最普遍的KL散度形式：

<img src="https://www.zhihu.com/equation?tex=-\epsilon \leq D(Y \| Z)=\mathbb{E}_{y \sim Y}\left[\ln \frac{\operatorname{Pr}[Y=y]}{\operatorname{Pr}[Z=y]}\right]\leq \epsilon
" alt="-\epsilon \leq D(Y \| Z)=\mathbb{E}_{y \sim Y}\left[\ln \frac{\operatorname{Pr}[Y=y]}{\operatorname{Pr}[Z=y]}\right]\leq \epsilon
" class="ee_img tr_noresize" eeimg="1">


## ABSTRACT

联合学习(FL)能够在**不违反隐私要求**的情况下跨多个客户端学习共享模型。但**资源和数据的异构性**对训练时间和模型精度有很大的影响。

为此提出了TIFL系统，即**基于层次**的联邦学习系统，该系统根据客户的**训练性能**将客户划分为多个层次，并在**每轮训练中从同一层次中选择客户**，以缓解由于资源和数据量的异构性而造成的掉队问题。

为了进一步训练nonIID(独立和相同分布)数据异质性，TiFL采用一种**自适应层选择方法**，根据观察到的训练性能和准确性实时更新分层。

我们在遵循谷歌FL架构的FL试验台上构建了TFL原型，并使用最先进的FL基准对其进行了评估。在各种不同的非均匀条件下，TIFL的性能都优于传统的FL。在提出的自适应层选择策略下，我们证明了TIFL在获得相同或更好的测试准确率的同时，获得了更快的训练性能。

---



## 1 INTRODUCTION

- **<u>*背景*</u>**

传统的高性能计算(HPC)中，所有数据都集中在一个位置，由拥有数百到数千个计算节点的超级计算机进行处理。出于安全和隐私方面的考虑，**GDPR（一般数据保护条例）和HIPAA（健康保险可携带性和责任法）**阻止了数据传输到集中的位置。

FL：在每个客户端(数据方)的本地数据上训练局部模型，并使用中央聚合器来累积局部模型的学习梯度来训练全局模型。虽然单个客户端的计算资源可能远不如传统超级计算机中的计算节点强大，但是来自大量客户端的**计算能力可以累积**起来，形成一个非常强大的”**分散式虚拟超级计算机**“。

FL的训练可以分为两种：跨设备FL和跨仓FL（cross-device和cross-silo）

1. 在跨设备FL中，客户端通常是具有**各种计算和通信能力**的大量移动或物联网设备；
2. 在跨仓FL中，客户端是具有**充足计算能力和可靠通信**的少数组织。

- **<u>*异质性情况*</u>**

本文中，我们关注的是跨设备FL(以下简称FL)，它本质上将**计算和通信资源的异构性推到了数据中心分布式学习**。

我们进行了一个案例研究，以量化客户中的数据和资源异构性如何影响FL的训练性能和模型准确性：

1. 训练吞吐量通常受**慢速客户端**(也称为掉队者)具有较小的计算能力和/或较慢的通信速度，我们称之为**<u>*资源异构性*</u>**。在数据中心分布式学习中，通常使用异步培训来缓解这一问题。但FL为了保护隐私，都是建立在同步的基础上的。
2. 不同的客户可能会在每轮训练中训练**不同的样本量**，导致不同的轮次时间（类似于掉队效应），成为**<u>*数据量异质性*</u>**。
3. 在FL中，数据类别和特征的分布取决于数据所有者，因此导致数据分布不均匀，称为不同的独立分布，此为**<u>*非IID数据异构性*</u>**。

资源异构性和数据量异质性信息可以在测量的训练时间中得到反映，但非IID数据异构性信息很难捕获。因为**任何测量类别和特征分布的尝试都违反了隐私保护要求**。

- **<u>*TIFL系统*</u>**

TIFL系统：

这里的关键思想是自适应地选择**每轮训练时间相似的客户**，以便在不影响模型精度的情况下缓解异构性问题。

具体地说，我们首先使用一个**轻量级分析器**来测量每个客户端的培训时间，并根据***测量的延迟将它们分组到不同的逻辑数据池中，称为层***。在每一轮训练中，基于TIFL的自适应客户端选择算法，从**同一层中随机选择均匀的客户端**。通过这种方式，由于属于同一层的客户端具有相似的训练时间，因此缓解了前两种异构性问题。

对于非IID数据，TIFL提出了一种自适应的客户端选择算法，该算法***以准确度为间接度量***来推断非IID数据的异构信息，并动态调整分层算法，使训练时间和准确率的影响降到最低。

---



## 2 RELATED WORK

在FL中，用户之间的数据迁移受到严格限制。FL不允许客户之间的信息共享。

FedCS建议通过基于截止日期的方法来解决客户选择问题，这种方法可以过滤出响应速度慢的客户。
然而，FedCS没有考虑这种方法如何影响模型培训中落后客户的贡献因素。

异步训练是解决数据中心分布式学习中掉队问题的常用方法，但几乎所有现有的隐私方法都是建立在同步模型权重更新的假设基础上的，所以异步训练很难应用到FL中

---



## 3 HETEROGENEITY IMPACT STUDY（异质性影响研究）

跨设备学习的关键特征之一是客户端之间显著的**资源和数据异构性**，这可能会影响训练吞吐量和模型精度。

- 资源异构性是训练过程中涉及的具有不同计算和通信能力的大量计算设备的结果。

- 数据异构性的产生主要有两个原因：(1)每个客户端可用的训练数据样本数量不同；(2)客户端之间的类别和特征分布不均匀。

---



### 3.1 Formulating Vanilla Federated Learning（构建香草联邦学习）传统FL

客户端池的总数为 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 为每一轮选择的客户端集合，每一轮整体训练从 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 中选择 <img src="https://www.zhihu.com/equation?tex=C_r" alt="C_r" class="ee_img tr_noresize" eeimg="1"> 个客户。

> <img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211211144903615.png" alt="image-20211211144903615" style="zoom:50%;" />
>
> 传统的FL算法如上：
>
> 1、初始化权重 <img src="https://www.zhihu.com/equation?tex=w_0" alt="w_0" class="ee_img tr_noresize" eeimg="1"> ；
>
> 2、每轮 <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1"> 开始时，聚合器将当前模型权重发送到随机选择的客户端的子集。
>
> 3、并行本地训练， <img src="https://www.zhihu.com/equation?tex=S_c" alt="S_c" class="ee_img tr_noresize" eeimg="1"> 为每个单独设备的数据量；
>
> 4、整体聚合

甚至有一种方法，**协调器**（服务器）设立了**主聚合器和子聚合器**，用于拓展和伸缩性。

---



### 3.2 Heterogeneity Impact Analysis（异构性影响分析）

在跨设备FL过程中，所涉及的客户端之间的资源和数据异构性可能导致不同的响应延迟（即，客户端接收训练任务和返回结果之间的时间），这通常被称为掉队问题。

每一轮的时间取决于最大时间，其中客户 <img src="https://www.zhihu.com/equation?tex=c_i" alt="c_i" class="ee_img tr_noresize" eeimg="1"> 对应的响应时延为 <img src="https://www.zhihu.com/equation?tex=L_i" alt="L_i" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=L_r" alt="L_r" class="ee_img tr_noresize" eeimg="1"> 为轮次 <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1"> 的响应时延：

<img src="https://www.zhihu.com/equation?tex=L_{r}=\operatorname{Max}\left(L_{1}, L_{2}, L_{3}, L_{4} \ldots L_{|C|}\right)
" alt="L_{r}=\operatorname{Max}\left(L_{1}, L_{2}, L_{3}, L_{4} \ldots L_{|C|}\right)
" class="ee_img tr_noresize" eeimg="1">
我们定义 <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1"> 为层数，每一层中客户端有相似的响应时延，设一共有 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 层， <img src="https://www.zhihu.com/equation?tex=\tau_m" alt="\tau_m" class="ee_img tr_noresize" eeimg="1"> 为最慢的一层， <img src="https://www.zhihu.com/equation?tex=\mid \tau_m \mid" alt="\mid \tau_m \mid" class="ee_img tr_noresize" eeimg="1"> 为该层的客户端数量。在3.1的传统算法中，所选择的客户端跨越多个层。

则除去最慢层后再选取 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 个客户端的概率表示：

<img src="https://www.zhihu.com/equation?tex=Pr=\frac{\left(\begin{array}{c}
|K|-\left|\tau_{m}\right| \\
|C|

\end{array}\right)}{\left(\begin{array}{l}
|K| \\
|C|

\end{array}\right)}
" alt="Pr=\frac{\left(\begin{array}{c}
|K|-\left|\tau_{m}\right| \\
|C|

\end{array}\right)}{\left(\begin{array}{l}
|K| \\
|C|

\end{array}\right)}
" class="ee_img tr_noresize" eeimg="1">
则 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 中至少有一个设备来自 <img src="https://www.zhihu.com/equation?tex=\tau_m" alt="\tau_m" class="ee_img tr_noresize" eeimg="1"> 的概率为：

<img src="https://www.zhihu.com/equation?tex=P r_{s}=1-P r=1-\frac{\left(\begin{array}{c}
|K|-\left|\tau_{m}\right| \\
|C|

\end{array}\right)}{\left(\begin{array}{l}
|K| \\
|C|

\end{array}\right)}=\quad 1-\frac{\left(|K|-\left|\tau_{m}\right|\right) \ldots\left(|K|-\left|\tau_{m}\right|-|C|+1\right)}{|K| \ldots(|K|-|C|+1)}
" alt="P r_{s}=1-P r=1-\frac{\left(\begin{array}{c}
|K|-\left|\tau_{m}\right| \\
|C|

\end{array}\right)}{\left(\begin{array}{l}
|K| \\
|C|

\end{array}\right)}=\quad 1-\frac{\left(|K|-\left|\tau_{m}\right|\right) \ldots\left(|K|-\left|\tau_{m}\right|-|C|+1\right)}{|K| \ldots(|K|-|C|+1)}
" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex==1-\frac{|K|-\left|\tau_{m}\right|}{|K|} \ldots \frac{|K|-\left|\tau_{m}\right|-|C|+1}{|K|-|C|+1}
" alt="=1-\frac{|K|-\left|\tau_{m}\right|}{|K|} \ldots \frac{|K|-\left|\tau_{m}\right|-|C|+1}{|K|-|C|+1}
" class="ee_img tr_noresize" eeimg="1">

> 存在定理：

<img src="https://www.zhihu.com/equation?tex=> \frac{a-1}{b-1}<\frac{a}{b}\qquad 1<a<b
> " alt="> \frac{a-1}{b-1}<\frac{a}{b}\qquad 1<a<b
> " class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\operatorname{Pr}_{s}>1-\frac{|K|-\left|\tau_{m}\right|}{|K|} \ldots \frac{|K|-\left|\tau_{m}\right|}{|K|}
=\quad 1-\left(\frac{|K|-\left|\tau_{m}\right|}{|K|}\right)^{|C|}
" alt="\operatorname{Pr}_{s}>1-\frac{|K|-\left|\tau_{m}\right|}{|K|} \ldots \frac{|K|-\left|\tau_{m}\right|}{|K|}
=\quad 1-\left(\frac{|K|-\left|\tau_{m}\right|}{|K|}\right)^{|C|}
" class="ee_img tr_noresize" eeimg="1">

现实生活中 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 的总值非常的大，故 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 的子集也非常大，则上式 <img src="https://www.zhihu.com/equation?tex=Pr_s\approx1" alt="Pr_s\approx1" class="ee_img tr_noresize" eeimg="1"> ，**说明从最慢的级别中选择至少一个客户端的概率非常高**。

---



### 3.3 Experimental Study（实验研究）

为了从实验方面说明论证资源异构性和数据量异构性的影响，实验如下：

1. 采用5组，每组4个客户端
2. 分别为五组配不同的CPU。（4个CPU、2个CPU、1个CPU、1/3个CPU、1/5个CPU）
3. 采用图像分类数据集CIFAR10进行FL训练
4. 对每个客户端进行不同数据大小的实验，得到数据异质性结果。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211212204516866.png" alt="image-20211212204516866" style="zoom:50%;" />

故引入TiFL，一种异构感知的客户端选择方法，在每一轮训练中选择最适合的客户端，在保持FL隐私特性的同时最小化异构性影响，从而提高跨设备FL的整体训练性能。

---



## 4 TIFL: A TIER-BASED FEDERATED LEARNING SYSTEM（一个基于层的联邦学习系统）

关键思想：每一轮学习都选择相近响应时延的设备端进行学习。

本节关键词：层选择方法（结果）、稻草人方案减轻异质性、自适应选择算法（解决稻草人的局限性）、通过一种层级选择模型模型可以利用层级选择概率和总训练轮数来估计期望训练时间

---



### 4.1 System Overview（系统概述）

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211212213832881.png" alt="image-20211212213832881" style="zoom:50%;" />

相比传统FL，其增加了两个新的模块：***分层模块和分层调度器***

操作步骤：

1. 通过轻量级分析收集所有可用客户机的延迟指标
2. 分层算法将进一步利用分析后的数据，这将客户机分组到称为“层”的单独逻辑池中。
3. 调度器选择一个层，然后从该层中随机选择目标数量的客户端。

由于TiFL是非侵入的，所以可以很好地加入到FL中并且不会触及到用户的隐私信息数据。

---



### 4.2 Profiling and Tiering（分级和分层）

操作步骤：

1. 所有的客户端初始化相应延迟 <img src="https://www.zhihu.com/equation?tex=L_i=0" alt="L_i=0" class="ee_img tr_noresize" eeimg="1"> ，执行配置轮 <img src="https://www.zhihu.com/equation?tex=sync_{rounds}" alt="sync_{rounds}" class="ee_img tr_noresize" eeimg="1"> 。
2. 聚合器要求客户端对本地数据进行训练，并等待 <img src="https://www.zhihu.com/equation?tex=T_{max}" alt="T_{max}" class="ee_img tr_noresize" eeimg="1"> 时间。
3. 在 <img src="https://www.zhihu.com/equation?tex=T_{max}" alt="T_{max}" class="ee_img tr_noresize" eeimg="1"> 时间内完成的 <img src="https://www.zhihu.com/equation?tex=RT_i" alt="RT_i" class="ee_img tr_noresize" eeimg="1"> 随实际训练时间递增，超时的按 <img src="https://www.zhihu.com/equation?tex=T_{max}" alt="T_{max}" class="ee_img tr_noresize" eeimg="1"> 递增。
4. 同步轮次完毕后，对于 <img src="https://www.zhihu.com/equation?tex=L_{i}>=sync_{rounds} * T_{max }" alt="L_{i}>=sync_{rounds} * T_{max }" class="ee_img tr_noresize" eeimg="1"> 的客户端实施丢弃。
5. 将训练延迟创建直方图，并分为 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 个组，同一组客户端为一层；计算每个组平均延迟，永久记录。
6. 定期执行以上操作，以适应通讯性能不断变化的系统。

---



### 4.3 Straw-man Proposal: Static Tier Selection Algorithm（静态层选择法）

上一节是分层，这一节说明如何在FL过程中从适当的层中选择客户，以提高训练性能。

要使选择更通用，可以指定根据预定义选择每个层𝑛𝑗的概率，所有层的概率总和为1。在每一层内，根据层内选择策略选择|𝐶|客户端。具体复杂的内层选择算法在后面再介绍。

- 真实世界中FL参与的设备很多，故层的数量设置为 <img src="https://www.zhihu.com/equation?tex=m<<|K|" alt="m<<|K|" class="ee_img tr_noresize" eeimg="1"> ，且每层的客户端数量 <img src="https://www.zhihu.com/equation?tex=n_j" alt="n_j" class="ee_img tr_noresize" eeimg="1"> 都比 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> ​要大。每层客户端数量的另一个考虑因素是，层中**太少的客户端可能会引入训练偏差**，因为这些客户端可能会被选择得太频繁，从而导致对这些客户端的数据过度拟合。解决此问题的一种方法是调整层选择概率。

- 但是，调整层选择概率会导致不同的权衡。如果用户的目标是减少整体培训时间，他们可能会增加选择速度较快的级别的机会。然而，仅从**最快的层抽取客户可能不可避免地会引入训练偏差**，因为不同的客户可能拥有分布在不同层的不同类型的训练数据集；因此，这种偏差可能最终会影响全局模型的准确性。

所以让来自不同层的客户端都参与进来，以便覆盖不同的训练数据集。

---



### 4.4 Adaptive Tier Selection Algorithm（自适应层选择）

上述简单的静态选择方法是直观的，但它没有提供一种自动调整权衡以优化训练性能的方法，也没有基于系统中的变化来调整选择。

故提出了一种自适应算法，该算法能够自动在**训练时间和准确率**之间取得平衡，并根据不断变化的系统条件自适应地调整训练轮次的选择概率。

上一节说道，过度的选择某一层会造成训练的过拟合（较快的层），所以需要其它层客户端的协调（较慢的层），问题是用哪种度量来平衡选择？由于目标是最小化训练模型的偏差，所以可以在整个训练过程中监控每一层的准确性。

若一轮学习中下来，发现层级 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 的准确性较低，**则说明层级 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 在上一轮次训练少，下一轮次应该作出更大的贡献**，基于这种思想，可以增加精确度较低的层的选择概率。为了获得良好的训练时间，我们还需要限制在多轮训练中选择速度较慢的级别。因此，我们引入了 <img src="https://www.zhihu.com/equation?tex=𝐶𝑟𝑒𝑑𝑖𝑡𝑠_t" alt="𝐶𝑟𝑒𝑑𝑖𝑡𝑠_t" class="ee_img tr_noresize" eeimg="1"> ，这是一个定义可以**选择某一层的次数的约束**。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211213145142104.png" alt="image-20211213145142104" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211213145204373.png" alt="image-20211213145204373" style="zoom:50%;" />

> - ***变量定义：***
>
> 1.  <img src="https://www.zhihu.com/equation?tex=Credits_t" alt="Credits_t" class="ee_img tr_noresize" eeimg="1"> ：层数 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 的信誉值，即被选择次数。
> 2.  <img src="https://www.zhihu.com/equation?tex=TestData_t" alt="TestData_t" class="ee_img tr_noresize" eeimg="1"> ：每个客户机上用来评估模型的数据
> 3.  <img src="https://www.zhihu.com/equation?tex=A_t^r" alt="A_t^r" class="ee_img tr_noresize" eeimg="1"> ：层 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 在第 <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1"> 轮上每一个客户机的平均测试精度。
> 4.  <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ：每过 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 轮次后更新层选择概率。（概率更新间隔）
> 5.  <img src="https://www.zhihu.com/equation?tex=NewProbs" alt="NewProbs" class="ee_img tr_noresize" eeimg="1"> ：层选择概率
> 6. 函数 <img src="https://www.zhihu.com/equation?tex=ChangeProbs" alt="ChangeProbs" class="ee_img tr_noresize" eeimg="1"> ：用于更新概率的函数
>
> - ***算法步骤：***
>
> 1. 初始化权重 <img src="https://www.zhihu.com/equation?tex=w_0" alt="w_0" class="ee_img tr_noresize" eeimg="1"> ，变量 <img src="https://www.zhihu.com/equation?tex=currentTier=1" alt="currentTier=1" class="ee_img tr_noresize" eeimg="1"> ，概率平均分配。
> 2. 先判断是否需要更新层选择概率：若此刻轮数整除 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> ，并且当前层精确度优于 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 轮之前的精确度，则按照精确度更新概率。
> 3. 每一轮次 <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1"> 都执行：当前层 <img src="https://www.zhihu.com/equation?tex=currentTier" alt="currentTier" class="ee_img tr_noresize" eeimg="1"> 根据 <img src="https://www.zhihu.com/equation?tex=NewProbs" alt="NewProbs" class="ee_img tr_noresize" eeimg="1"> 概率进行选拔，如果其 <img src="https://www.zhihu.com/equation?tex=Credits_t" alt="Credits_t" class="ee_img tr_noresize" eeimg="1"> 值大于0则入选，并且将其减1。
> 4. 在选中的层内执行传统FL计算梯度聚合。
> 5. 更新每一轮中每一层内的准确度 <img src="https://www.zhihu.com/equation?tex=A_t^r" alt="A_t^r" class="ee_img tr_noresize" eeimg="1"> 。
>
> - ***概率更新函数：***
>
> 1. 将每一层的平均精确度 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 进行升序排列。
> 2. 计算信用额度不为0的所有层总和： <img src="https://www.zhihu.com/equation?tex=D=n *(n-1) / 2" alt="D=n *(n-1) / 2" class="ee_img tr_noresize" eeimg="1"> 
> 3.  <img src="https://www.zhihu.com/equation?tex=N e w \operatorname{Probs}[t]=(n-i) / D" alt="N e w \operatorname{Probs}[t]=(n-i) / D" class="ee_img tr_noresize" eeimg="1"> 

---



### 4.5 Training Time Estimation Model（训练时间模型）

在现实生活场景中，培训时间和资源预算通常是有限的。FL需要在**训练时间和准确性之间进行折衷**。训练时间估计模型将有助于用户在**训练时间-精度**的权衡曲线上导航，从而有效地实现预期的训练目标。

因此，我们建立了一个训练时间估计模型，该模型可以根据给定的延迟值和各层的选择概率来估计总训练时间:

<img src="https://www.zhihu.com/equation?tex=L_{a l l}=\sum_{i=1}^{n}\left(\max \left(L_{t i e r_{-} i}\right) * P_{i}\right) * R
" alt="L_{a l l}=\sum_{i=1}^{n}\left(\max \left(L_{t i e r_{-} i}\right) * P_{i}\right) * R
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=L_{t i e r_{-} i}" alt="L_{t i e r_{-} i}" class="ee_img tr_noresize" eeimg="1"> 是第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 层的响应延迟， <img src="https://www.zhihu.com/equation?tex=P_i" alt="P_i" class="ee_img tr_noresize" eeimg="1"> 是第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 层被选中的概率， <img src="https://www.zhihu.com/equation?tex=R" alt="R" class="ee_img tr_noresize" eeimg="1"> 是训练总轮数，可以看做每一层的**延迟期望与轮数相乘**。

---



### 4.6 Discussion: Compatibility with Privacy-Preserving Federated Learning（与隐私保护联合学习的兼容性）

隐私保护可以通过每个客户实现一个集中的私人学习算法作为他们的本地培训方法来实现。例如每个客户端将**适当的噪声**添加到他们的本地学习中，以保护其个人数据集的隐私。

具体看下具备知识。𝜖限定了任何个体可能对算法输出产生的影响，而𝛿定义了违反该界限的概率。

较小的𝜖值意味着更严格的界限，并提供**更强的隐私保障**，但需要向客户端发送到FL服务器的模型更新添加**更多噪声**，这会导致模型不太准确。

每个客户端在每次回复查询时都会向其模型更新添加不同的隐私噪声，并且此噪声可防止任何隐私信息泄露。在我们的方法中，根据基于层的随机过程来查询客户端，其中客户端被选择的次数既取决于**层选择策略，也取决于层内选择策略**。由于差别隐私的组成特性以及**每个客户添加自己的噪声**的事实，每个客户可以**监视和控制他们的差别隐私预算是如何花费**的，从而可以控制他们的差别隐私保障。因此，每个客户都可以通过**停止回复**来满足他们的隐私要求，如果他们的预算已经被消耗了。

总的来说，当前方式与常见的隐私保护方式可以很好很方便地结合起来。

---



## 5 EXPERIMENTAL EVALUATION（实验评价）

用原始方法和自适应方式分别构建了TiFL，在资源异构、数据异构和资源加数据异构三种场景下进行了大量实验。

### 5.1 Experimental Setup（实验装置）

装置：在一个CPU集群上分配了50个客户端，在每一轮训练中，选择5个客户端对自己的数据进行训练。在我们的原型中，我们简化了系统，使用一个功能强大的**单一聚合器**。

此外还采用了LEAF框架。

> LEAF是联邦学习的开源基准。它包括(1)一组开源数据集，(2)一组统计和系统指标，(3)一组参考实现。

LEAF本质上为非IID提供了数据量和类分布的异构性，但不提供客户端之间的资源异构性。

对于资源异构性，每个客户端的这种资源分配是通过**均匀随机分布**完成的，从而导致每个硬件类型的客户端数量相等。

对于LEAF，直接采用框架提供的数据集，对于Cifar10/MNIST/FMNIST，我们设置每个客户端的非iid级别，进行数据分发后再随机分配到一个节点。

---



### 5.2 Experimental Results（实验结果）

- ***数据集***

MNIST和Fashion-MNIS：每个包含60,000个训练图像和10,000个测试图像

Cifar10：总共有60,000张彩色图像，划分为5万张训练图像和1万张测试图像。

采用了LEAF中的FEMNIST数据集：62类组成的图像分类数据集，该数据集本质上是非iid的，数据量和类分布具有异质性。

- ***训练超参数***

1. 使用RMSprop作为局部训练的优化器，RMSprop作为局部训练的优化器，设置初始学习率(𝜂)为0.01%，衰减率为0.995。每个客户端的本地批量为10，本地历元为1。**CIFAR10的客户端总数**(|𝐾|)为50个，每轮参与的客户端(|𝐶|)为5个。

2. FEMNIST的客户端总数为182个，每轮客户端为18个，默认训练参数由LEAF框架提供(SGD，LR为0.004，批大小为10)。

- **资源异构**

一共分为5组。

对于MNIST和Fashion-MNIST，每组分别分配2个CPU、1个CPU、0.75个CPU、0.5个CPU和0.25个CPU。对于更大的Cifar10和女权主义型号，每组分别分配4个CPU、2个CPU、1个CPU、0.5个CPU和0.1个CPU。

- **数据异构性**

1. 数量异构性：训练数据样本分布分别为不同分组总数据集的10%、15%、20%、25%、30%。

2. 非IID数据：对于MNIST和Fashion-MNIST，首先按值对标签进行排序，平均划分为100个分片，然后为每个客户端分配两个分片。

   对于Cifar10，我们以类似的方式不均匀地分割数据集，最后限定每个客户端的类数为5。

- **计划策略**

称号：***vanilla***：从客户端的中随机选择5个客户端；***fast***：每一轮只选择TiFL中最快的客户端；***random***：演示了最快层的选择优先于较慢层的情况。***uniform***：每一层被选中的概率都相等；***slow***：选择最慢的层

我们将上述政策用于CIFAR-10和FEMINST。

对于MNIST和Fashion-MNIST，考虑到它是一个轻量级得多的工作负载，重点展示了当策略更积极地向FAST层（Fast1到Fast3），最慢层的选择概率从0.1降到0，而所有其他层的选择概率相同。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214155538740.png" alt="image-20211214155538740" style="zoom:50%;" />

#### 5.2.1 Training Time Estimation via Analytical Model（通过模型进行训练时间估计）

通过将模型的估计结果与试验台实验的测量结果进行比较，来评估我们的训练时间估计模型在不同朴素层选择策略上的准确性。

利用**每层的平均延迟、选择概率和训练轮次总数**，以估计训练时间。我们使用平均预测误差(MAPE)作为评估指标，其定义如下：

<img src="https://www.zhihu.com/equation?tex=\text { MAPE }=\frac{\left|L_{all }^{e s t}-L_{all}^{a c t}\right|}{L_{all }^{act}} * 100
" alt="\text { MAPE }=\frac{\left|L_{all }^{e s t}-L_{all}^{a c t}\right|}{L_{all }^{act}} * 100
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=L_{all }^{e s t}" alt="L_{all }^{e s t}" class="ee_img tr_noresize" eeimg="1"> 是由估计模型计算的估计训练时间， <img src="https://www.zhihu.com/equation?tex=L_{all}^{a c t}" alt="L_{all}^{a c t}" class="ee_img tr_noresize" eeimg="1"> 是实际时间。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214162054843.png" alt="image-20211214162054843" style="zoom:50%;" />

说明模型具有很高的精度，由于系统具有随机性，所以误差还是会稍有变化。

---

#### 5.2.2 Resource Heterogeneity（资源异质性）

从**训练时间和模型准确性**进行分析，并且假设没有数据异构性。只是为了单独展示TiFL如何单独地训练资源异构性。

在此只展示Cifar10的结果，因为MNIST和Fashion-MNIST有相似的观测结果。

> <img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214165310838.png" alt="image-20211214165310838" style="zoom:50%;" />
>
> 图（a）：fast很快没的说，uniform也很快，是因为每一轮选择都是在同组中选择，但之前有很大概率每一轮都选到最慢的组。
>
> 图（c）：训练轮数相同的情况下，说明大家效果都差不太多；
>
> 图（e）：训练时间一定的话，根据选择层的概率不同，差异很大。也就是说在相同时间内各种基准集训练的轮次并不相同。

---

#### 5.2.3 Data Heterogeneity（数据异构性）

- ***数据量异构***

在本小节只分析数据异构的情况，为每个客户端都分配2个CPU。在此依旧只展示Cifar10的结果。

> 图（b）：明显比（a）要快，是因为数据量的异构性也可能导致轮次时间的不同。
>
> 图（d）：slow的准确性最高，因为其包含30%的数据，但训练时间最差
>
> 图（f）：其中fast最快，因为第一层是fast，只包含10%的数据

说明数据异构性在TiFL上表现也很良好。但实际情况下图（d）和图（f）情况不容易出现，因为数据量足够多。

- ***数据类别异构***

经过观察类别异构对时间影响不大，但对准确性有影响。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214192902390.png" alt="image-20211214192902390" style="zoom:50%;" />

分别用随机分布、2类、5类、10类进行测试，发现对所有类别的影响是一致的。

---

#### 5.2.4 Resource and Data Heterogeneity（资源和数据异构性）

本节讨论的最接近实际情况，只讨论静态选择的结果。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214193823355.png" alt="image-20211214193823355" style="zoom:50%;" />

> MNIST 和 Fashion-MNIST数据集如上图两列
>
> 图（a，b）：fast级别越高则培训时间越短，训练速度更快
>
> 图（c，d）：准确性fast3最低，因为完全忽略了第五层的数据

下图左边一列是Cifar10具有资源异构和IID异构的情况，右边一列是Cifar10具有资源、数据量、IID异构的情况。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211214195812493.png" alt="image-20211214195812493" style="zoom:50%;" />

> 左边一列表现出，时间上和资源异构性差不太多，因为非IID的情况在训练时不会产生过大影响，但准确性**<u>*都*</u>**有所下降。因为不同类别之间的训练偏差过大。
>
> 右边一列为三种异质性全部应用的结果，图（b）与（a）区别不大，原因是不同数据量的训练时间可以通过“层”的设定校正***（有cpu高的但数据量多，有cpu少的但数据量少，有可能被分到同一组）***
>
> 然而图（d）显示精度大不相同。fast明显变小，是因为其数据量异质性放大了差异，***即某些类的数据变得非常少甚至没有***。
>
> 图（F）显示了时间精度，可以看到应用TiFL时间上显著提高。

---

#### 5.2.5 Adaptive Selection Policy（适应性选择策略）

TIFL中的朴素选择方法可以显著提高训练时间，但有时会出现准确率不高的情况，特别是在数据异构性很强的情况下。**在此将自适应策略和均匀策略作比较，因为均匀策略在静态中准确度最高**。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211215092736042.png" alt="image-20211215092736042" style="zoom:50%;" />

> 训练时间上：数据异构和类别异构两种情况，TiFL都好，但结合起来其与uniform相比略高一点点，但时间上还是达到了随机算法的一半，且达到了与普通算法相当的精度。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211215094750927.png" alt="image-20211215094750927" style="zoom:50%;" />

> 不同策略在非IID情况下的准确性，可以看到TiFL是较优的。

---

#### 5.2.6 Adaptive Selection Policy（LEAF）

用LEAF框架进行配置，配置了182个客户端，分配了相应的资源，进一步将TFL的分层模块和选择策略整合到扩展的LEAF框架中。

将层的总数限制为5层，并且在每轮期间我们选择10个客户端，每轮1个epoch。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/TFL：一种基于层次的联邦学习系统/image-20211215102742601.png" alt="image-20211215102742601" style="zoom:50%;" />

> FAST准确率最低的原因是第1层中客户端的训练点较少。在训练时间方面，快速和随机都优于自适应
>
> 准确性方面自适应优于静态情况。

与传统FL相比，TiFL在整体训练时间上提高了3倍，准确率提高了6%。