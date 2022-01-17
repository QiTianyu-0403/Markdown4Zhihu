# Optimizing Federated Learning on Non-IID Data with Reinforcement Learning学习笔记

基于强化学习的非IID数据联合学习优化

[toc]

## 预备知识

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20220110205631094.jpg" alt="image-20220110205631094" style="zoom:50%;" />

## Abstract

由于移动设备的**网络连接**有限，联合学习在所有参与设备上并行执行模型更新和聚合是不切实际的。
此外，所有设备上的数据样本通常**不是独立且相同分布**的(IID)，这给联合学习的收敛和速度带来了额外的挑战。

我们提出了一种体验驱动的控制框架--FAVOR，它智能地选择客户端设备参与每一轮联合学习，以抵消非IID数据带来的偏差，并加快收敛速度。我们观察到设备上的训练数据分布与基于这些数据训练的模型权重之间存在隐性联系，这使得我们能够根据上传的模型权重来描述该设备上的数据分布。然后，我们提出了一种基于DQN的机制，学习在每个交流回合中选择设备子集，以最大化奖励，鼓励验证准确性的提高，惩罚使用更多的交流回合。

## INTRODUCTION

移动设备上的分布式机器学习面临着一些基本挑战，例如无线网络的有限连接，移动设备的不稳定可用性，以及难以在统计上描述[1]-[4]的本地数据集的非iid分布。因此，联邦学习中常见的做法是，每轮模型训练中随机选取设备的子集，以避免由于网络条件不稳定和设备分散造成的长尾等待时间。

然而，现有的联邦学习方法还没有解决异构局部数据集带来的统计挑战。由于不同的用户有不同的设备使用模式，位于任何单个设备上的数据样本和标签可能遵循不同的分布，这不能代表全局数据分布。对于FedAvg，随机选择的数据子集可能不能反映全局视图下的真实数据分布。

本文提出的FAVOR数据框架，旨在通过**智能选择设备**来提高联邦学习的性能。**基于强化学习，FAVOR的目的是通过学习主动选择每个通信轮中设备的最佳子集，来加速和稳定联邦学习过程，以抵消非iid数据引入的偏差。**

## BACKGROUND AND MOTIVATION

我们演示了如何在每一轮正确选择客户端设备，以提高非IID数据上的联合学习性能。

### A.Federated Learning

这里将FL看做一个 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 分类问题，特征空间 <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> ，标签空间 <img src="https://www.zhihu.com/equation?tex=Y=[C]" alt="Y=[C]" class="ee_img tr_noresize" eeimg="1"> ，令 <img src="https://www.zhihu.com/equation?tex=(x,y)" alt="(x,y)" class="ee_img tr_noresize" eeimg="1"> 为特定的标记样本，其中 <img src="https://www.zhihu.com/equation?tex=f: \mathcal{X} \rightarrow \mathcal{S}" alt="f: \mathcal{X} \rightarrow \mathcal{S}" class="ee_img tr_noresize" eeimg="1"> 是影射，其中 <img src="https://www.zhihu.com/equation?tex=\mathcal{S}=\left\{\boldsymbol{z} \mid \sum_{i=1}^{C} z_{i}=1, z_{i} \geqslant 0, \forall i \in[C]  \right\}" alt="\mathcal{S}=\left\{\boldsymbol{z} \mid \sum_{i=1}^{C} z_{i}=1, z_{i} \geqslant 0, \forall i \in[C]  \right\}" class="ee_img tr_noresize" eeimg="1"> ，也就是说，向量值函数 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为每个样本 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 产生概率向量 <img src="https://www.zhihu.com/equation?tex=z" alt="z" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=f_i" alt="f_i" class="ee_img tr_noresize" eeimg="1"> 预测样本属于第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 类的概率。

设 <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 表示权重，则训练所用交叉熵训练表示为：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\ell(\boldsymbol{w}) &:=\mathbb{E}_{\boldsymbol{x}, y \sim p}\left[\sum_{i=1}^{C} \mathbb{1}_{y=i} \log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right] \\
&=\sum_{i=1}^{C} p(y=i) \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]
\end{aligned}
" alt="\begin{aligned}
\ell(\boldsymbol{w}) &:=\mathbb{E}_{\boldsymbol{x}, y \sim p}\left[\sum_{i=1}^{C} \mathbb{1}_{y=i} \log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right] \\
&=\sum_{i=1}^{C} p(y=i) \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
学习问题是要解决以下优化问题：

<img src="https://www.zhihu.com/equation?tex=\operatorname{minimize}_{\boldsymbol{w}} \sum_{i=1}^{C} p(y=i) \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]
" alt="\operatorname{minimize}_{\boldsymbol{w}} \sum_{i=1}^{C} p(y=i) \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]
" class="ee_img tr_noresize" eeimg="1">
总共有 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 个客户端，第 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 个设备有 <img src="https://www.zhihu.com/equation?tex=m^{(k)}" alt="m^{(k)}" class="ee_img tr_noresize" eeimg="1"> 个样本，服从数据分布 <img src="https://www.zhihu.com/equation?tex=p^{(k)}" alt="p^{(k)}" class="ee_img tr_noresize" eeimg="1"> ，每一轮学习中，下载当前全局模型权重 <img src="https://www.zhihu.com/equation?tex=w_{t-1}" alt="w_{t-1}" class="ee_img tr_noresize" eeimg="1"> ，则在本地进行SGD训练表示为：

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{w}_{t}^{(k)}=\boldsymbol{w}_{t-1}-\eta \nabla \ell\left(\boldsymbol{w}_{t-1}\right)=\boldsymbol{w}_{t-1}-\eta \sum_{i=1}^{C} p^{(k)}(y=i) \nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}\left(\boldsymbol{x}, \boldsymbol{w}_{t-1}\right)\right]
" alt="\boldsymbol{w}_{t}^{(k)}=\boldsymbol{w}_{t-1}-\eta \nabla \ell\left(\boldsymbol{w}_{t-1}\right)=\boldsymbol{w}_{t-1}-\eta \sum_{i=1}^{C} p^{(k)}(y=i) \nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}\left(\boldsymbol{x}, \boldsymbol{w}_{t-1}\right)\right]
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\boldsymbol{w}_{t}^{(k)}" alt="\boldsymbol{w}_{t}^{(k)}" class="ee_img tr_noresize" eeimg="1"> 可以是本地一个或多个阶段后的结果，结束后，每个设备向服务器汇报一个模型权重的残差：

<img src="https://www.zhihu.com/equation?tex=\Delta_{t}^{(k)}:=\boldsymbol{w}_{t}^{(k)}-\boldsymbol{w}_{t-1}^{(k)}
" alt="\Delta_{t}^{(k)}:=\boldsymbol{w}_{t}^{(k)}-\boldsymbol{w}_{t-1}^{(k)}
" class="ee_img tr_noresize" eeimg="1">
也可以直接传输 <img src="https://www.zhihu.com/equation?tex=\boldsymbol{w}_{t}^{(k)}" alt="\boldsymbol{w}_{t}^{(k)}" class="ee_img tr_noresize" eeimg="1"> ，服务器接受之后，执行**联邦学习**：

<img src="https://www.zhihu.com/equation?tex=\begin{gathered}
\Delta_{t}=\sum_{k=1}^{K} m^{(k)} \Delta_{t}^{(k)} / \sum_{k=1}^{K} m^{(k)} \\
\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}+\Delta_{t}
\end{gathered}
" alt="\begin{gathered}
\Delta_{t}=\sum_{k=1}^{K} m^{(k)} \Delta_{t}^{(k)} / \sum_{k=1}^{K} m^{(k)} \\
\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}+\Delta_{t}
\end{gathered}
" class="ee_img tr_noresize" eeimg="1">

### B.The Challenges of Non-IID Data Distribution

每个设备上拥有的数据通常是IID的。当数据分布是非iid时，FedAvg是不稳定的，甚至可能发散。

局部执行的SGD算法，其目标是最小化每个设备上 <img src="https://www.zhihu.com/equation?tex=m^{(k)}" alt="m^{(k)}" class="ee_img tr_noresize" eeimg="1"> 个局部样本的损失值，但全局目标是最小化 <img src="https://www.zhihu.com/equation?tex=\sum_{k=1}^{K} m^{(k)}" alt="\sum_{k=1}^{K} m^{(k)}" class="ee_img tr_noresize" eeimg="1"> 的损失值。随着我们不断将不同设备上的模型拟合到异构的局部数据中，这些局部模型的权值 <img src="https://www.zhihu.com/equation?tex=w^{(k)}" alt="w^{(k)}" class="ee_img tr_noresize" eeimg="1"> 之间的分歧会不断累积，最终导致学习性能下降。

实验证明，**如果不是随机选择的话，而是使用聚类算法选择设备，**可以帮助均衡数据分布，加快收敛速度。

为此做了一个实验，用CNN训练MNIST，对于非IID的构造：其80%数据来自一个类别，其他20%来自其他类别。一共100台设备，每一轮随机抽取10台设备进行训练，看到同样达到99%的准确性，FedAvg在Non-IID用时极长：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211218165005566.png" alt="image-20211218165005566" style="zoom:50%;" />

所以需要调整设备选择策略，由于隐私的原因，不能窥探每个设备的数据分布，我们观察到在**设备上的数据分布和在该设备上训练的模型权重之间存在隐含的联系**。因此，我们可以通过分第一轮之后的局部模型权重来预测每个设备。

设 <img src="https://www.zhihu.com/equation?tex=w_{init}" alt="w_{init}" class="ee_img tr_noresize" eeimg="1"> 为初始权重，则经过第一轮的梯度下降计算后可以得到：

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{w}_{1}^{(k)}=\boldsymbol{w}_{i n i t}-\eta \sum_{i=1}^{C} p^{(k)}(y=i) \nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}\left(\boldsymbol{x}, \boldsymbol{w}_{i n i t}\right)\right]
" alt="\boldsymbol{w}_{1}^{(k)}=\boldsymbol{w}_{i n i t}-\eta \sum_{i=1}^{C} p^{(k)}(y=i) \nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}\left(\boldsymbol{x}, \boldsymbol{w}_{i n i t}\right)\right]
" class="ee_img tr_noresize" eeimg="1">
两个设备之间的权重差，利用某一种界定技术，可以推导出第一轮梯度算法**权重发散的界限**：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\left\|\boldsymbol{w}_{1}^{\left(k^{\prime}\right)}-\boldsymbol{w}_{1}^{(k)}\right\| \leqslant 
\eta g_{\max }\left(\boldsymbol{w}_{i n i t}\right) \sum_{i=1}^{C}\left\|p^{\left(k^{\prime}\right)}(y=i)-p^{(k)}(y=i)\right\|

\end{aligned}
" alt="\begin{aligned}
\left\|\boldsymbol{w}_{1}^{\left(k^{\prime}\right)}-\boldsymbol{w}_{1}^{(k)}\right\| \leqslant 
\eta g_{\max }\left(\boldsymbol{w}_{i n i t}\right) \sum_{i=1}^{C}\left\|p^{\left(k^{\prime}\right)}(y=i)-p^{(k)}(y=i)\right\|

\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
其中：

<img src="https://www.zhihu.com/equation?tex=g_{\max }(\boldsymbol{w}):=\max _{i=1}^{C}\left\|\nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]\right\|

" alt="g_{\max }(\boldsymbol{w}):=\max _{i=1}^{C}\left\|\nabla_{\boldsymbol{w}} \mathbb{E}_{\boldsymbol{x} \mid y=i}\left[\log f_{i}(\boldsymbol{x}, \boldsymbol{w})\right]\right\|

" class="ee_img tr_noresize" eeimg="1">
这说明了即使使用相同的初始权重训练本地模型，在设备 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=k^{\prime}" alt="k^{\prime}" class="ee_img tr_noresize" eeimg="1"> 上的数据分布的差异存在隐含的联系。

基于以上分析，我们首先让100台设备中的每一台下载相同的初始全局权重(随机生成)，并基于本地数据执行SGD的一个epoch以得到 <img src="https://www.zhihu.com/equation?tex=w_1^{(k)},k=1,...,100" alt="w_1^{(k)},k=1,...,100" class="ee_img tr_noresize" eeimg="1"> ，然后利用K-Center算法进行聚类，分成10组。每一轮还从10组中分别选一个设备进行训练，可以看到图中，经过K-Center之后的效果会更好。

### C.Deep Reinforcement Learning (DRL)

Q-learning算法：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
Q_{\pi}\left(\boldsymbol{s}_{t}, a\right): &=\mathbb{E}_{\pi}\left[\sum_{k=1}^{\infty} \gamma^{k-1} r_{t+k-1} \mid \boldsymbol{s}_{t}, a\right] 
=\mathbb{E}_{\boldsymbol{s}_{t+1}, a}\left[r_{t}+\gamma Q_{\pi}\left(\boldsymbol{s}_{t+1}, a\right) \mid \boldsymbol{s}_{t}, a_{t}\right]
\end{aligned}
" alt="\begin{aligned}
Q_{\pi}\left(\boldsymbol{s}_{t}, a\right): &=\mathbb{E}_{\pi}\left[\sum_{k=1}^{\infty} \gamma^{k-1} r_{t+k-1} \mid \boldsymbol{s}_{t}, a\right] 
=\mathbb{E}_{\boldsymbol{s}_{t+1}, a}\left[r_{t}+\gamma Q_{\pi}\left(\boldsymbol{s}_{t+1}, a\right) \mid \boldsymbol{s}_{t}, a_{t}\right]
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
最优行为值函数：

<img src="https://www.zhihu.com/equation?tex=Q^{*}\left(\boldsymbol{s}_{t}, a\right):=\mathbb{E}_{\boldsymbol{s}_{t+1}}\left[r_{t}+\gamma \max _{a} Q^{*}\left(\boldsymbol{s}_{t+1}, a\right) \mid \boldsymbol{s}_{t}, a\right]
" alt="Q^{*}\left(\boldsymbol{s}_{t}, a\right):=\mathbb{E}_{\boldsymbol{s}_{t+1}}\left[r_{t}+\gamma \max _{a} Q^{*}\left(\boldsymbol{s}_{t+1}, a\right) \mid \boldsymbol{s}_{t}, a\right]
" class="ee_img tr_noresize" eeimg="1">
利用深度学习DNN，RL学习问题变成最小化目标和近似器之间的均方误差损失，这被描述为：

<img src="https://www.zhihu.com/equation?tex=\ell_{t}\left(\boldsymbol{\theta}_{t}\right):=\left(r_{t}+\gamma \max _{a} Q\left(\boldsymbol{s}_{t+1}, a ; \boldsymbol{\theta}_{t}\right)-Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right)\right)^{2}
" alt="\ell_{t}\left(\boldsymbol{\theta}_{t}\right):=\left(r_{t}+\gamma \max _{a} Q\left(\boldsymbol{s}_{t+1}, a ; \boldsymbol{\theta}_{t}\right)-Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right)\right)^{2}
" class="ee_img tr_noresize" eeimg="1">

## DRL FOR CLIENT SELECTION

### A.The Agent based on Deep Q-Network

设总共N个设备，利用联邦学习训练一个目标精度为 <img src="https://www.zhihu.com/equation?tex=\Omega" alt="\Omega" class="ee_img tr_noresize" eeimg="1"> 的任务，选择 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 为子集的设备进行训练。

- 状态

第 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 轮的状态 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 表示为 <img src="https://www.zhihu.com/equation?tex=\boldsymbol{s}_{t}=\left(\boldsymbol{w}_{t}, \boldsymbol{w}_{t}^{(1)}, \ldots, \boldsymbol{w}_{t}^{(N)}\right)" alt="\boldsymbol{s}_{t}=\left(\boldsymbol{w}_{t}, \boldsymbol{w}_{t}^{(1)}, \ldots, \boldsymbol{w}_{t}^{(N)}\right)" class="ee_img tr_noresize" eeimg="1"> ，其中包括全局权重参数和所有设备本地的权重参数。

结果的状态空间可能是巨大的，例如，一个CNN模型可以包含数百万个权重。训练具有如此大状态空间的DQN具有挑战性。在实践中，我们建议在状态空间上应用一种有效的**轻量级降维技术**，即在模型权重上

- 动作

一般情况会从 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 个设备中选择 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 个设备，但操作空间过大，使RL训练复杂化，为此提出一个小技巧：

在训练DQN过程中，只选择1个设备训练，即 <img src="https://www.zhihu.com/equation?tex=a=\{1,2,3,...,N\}" alt="a=\{1,2,3,...,N\}" class="ee_img tr_noresize" eeimg="1"> 。

具体操作步骤是：训练DQN时，先按照一次选一个设备进行训练，得到相应的Q表（网络）。具体选择应用时，可以得到当前状态 <img src="https://www.zhihu.com/equation?tex=s_t" alt="s_t" class="ee_img tr_noresize" eeimg="1"> 下，N个动作选择的Q值（利用Q网络），选择Q值最大的前K个设备

- 奖励

我们设每轮结束后的奖励为： <img src="https://www.zhihu.com/equation?tex=r_{t}=\Xi^{\left(\omega_{t}-\Omega\right)}-1, t=1, \ldots, T" alt="r_{t}=\Xi^{\left(\omega_{t}-\Omega\right)}-1, t=1, \ldots, T" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=\omega_{t}" alt="\omega_{t}" class="ee_img tr_noresize" eeimg="1"> 表示全局模型的准确率， <img src="https://www.zhihu.com/equation?tex=\Omega" alt="\Omega" class="ee_img tr_noresize" eeimg="1"> 表示目标准确率， <img src="https://www.zhihu.com/equation?tex=\Xi" alt="\Xi" class="ee_img tr_noresize" eeimg="1"> 表示一个常数，确保 <img src="https://www.zhihu.com/equation?tex=\omega_{t}" alt="\omega_{t}" class="ee_img tr_noresize" eeimg="1"> 正常增长。其中 <img src="https://www.zhihu.com/equation?tex=0 \leq w_{t} \leq \Omega \leq 1 \text ,  r_{t} \in(-1,0]" alt="0 \leq w_{t} \leq \Omega \leq 1 \text ,  r_{t} \in(-1,0]" class="ee_img tr_noresize" eeimg="1"> 。

因此总奖励的计算公式是：

<img src="https://www.zhihu.com/equation?tex=R=\sum_{t=1}^{T} \gamma^{t-1} r_{t}=\sum_{t=1}^{T} \gamma^{t-1}\left(\Xi^{\left(\omega_{t}-\Omega\right)}-1\right)
" alt="R=\sum_{t=1}^{T} \gamma^{t-1} r_{t}=\sum_{t=1}^{T} \gamma^{t-1}\left(\Xi^{\left(\omega_{t}-\Omega\right)}-1\right)
" class="ee_img tr_noresize" eeimg="1">
对于 <img src="https://www.zhihu.com/equation?tex=r_t" alt="r_t" class="ee_img tr_noresize" eeimg="1"> 的解释，前一项 <img src="https://www.zhihu.com/equation?tex=\Xi^{\left(\omega_{t}-\Omega\right)}" alt="\Xi^{\left(\omega_{t}-\Omega\right)}" class="ee_img tr_noresize" eeimg="1"> 激励智能体能选择精度较高的设备，同时还控制奖励R的增长速度。随着机器学习的进行， <img src="https://www.zhihu.com/equation?tex=\omega_t" alt="\omega_t" class="ee_img tr_noresize" eeimg="1"> 的变化会越来越小，所以用指数来放大， <img src="https://www.zhihu.com/equation?tex=\Xi" alt="\Xi" class="ee_img tr_noresize" eeimg="1"> 在实验中取64。-1的目的是鼓励agent在更少的回合中完成训练，因为需要的回合越多，agent得到的累积奖励越少。

### B.Workflow

学习步骤如下：

1. 所有N个符合条件的设备都检入FL服务器。
2. 每个设备从服务器下载初始化好的权重 <img src="https://www.zhihu.com/equation?tex=w_{init}" alt="w_{init}" class="ee_img tr_noresize" eeimg="1"> ，并进行一个epoch的SGD学习，得到每个设备的模型权重 <img src="https://www.zhihu.com/equation?tex=\left\{\boldsymbol{w}_{1}^{(k)}, k \in[N]\right\}" alt="\left\{\boldsymbol{w}_{1}^{(k)}, k \in[N]\right\}" class="ee_img tr_noresize" eeimg="1"> ，并给到服务器。
3. 服务器收到client上传的模型之后更新服务器上的模型备份，并且对于每一个clienta，DQN计算其对应的值函数 <img src="https://www.zhihu.com/equation?tex=Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}\right)" alt="Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}\right)" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=a=1,2,...,N" alt="a=1,2,...,N" class="ee_img tr_noresize" eeimg="1"> 。
4. 选择值函数最大的K个设备，将全局模型下发，训练，得到 <img src="https://www.zhihu.com/equation?tex=\left\{\boldsymbol{w}_{t+1}^{(k)} \mid k \in[K]\right\}" alt="\left\{\boldsymbol{w}_{t+1}^{(k)} \mid k \in[K]\right\}" class="ee_img tr_noresize" eeimg="1"> 。
5. 通过FedAvg聚合方式得到全局的 <img src="https://www.zhihu.com/equation?tex=w_t" alt="w_t" class="ee_img tr_noresize" eeimg="1"> ，再继续重复3-5步骤。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211220141418436.png" alt="image-20211220141418436" style="zoom:50%;" />

若设备 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 有数据更新时，会请求全局模型的权值，并进行一个本地的epoch训练，生成新的全局权值，从而改变DQN的状态集合。

### C.Dimension Reduction（降维）

因为本文状态采用的是权重，所以数据量巨大。故使用PCA对模型进行压缩，用来表示状态。

在此，只用第一轮epoch得到的权重 <img src="https://www.zhihu.com/equation?tex=\left\{\boldsymbol{w}_{1}^{(k)}, k \in[N]\right\}" alt="\left\{\boldsymbol{w}_{1}^{(k)}, k \in[N]\right\}" class="ee_img tr_noresize" eeimg="1"> 进行PCA的计算，之后的迭代都通过该模型进行线性变换进行主成分分析，从而节省开销。

为此做了一个实验，利用CNN训练MNIST，一共具有431080个维度的数据。采用的数据类别分配方式和第二节一样：80%相同的数据。利用第一次迭代后的权值，将其投影到二维空间主成分上，可以看到能够按照其权重分为10类。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211220152847894.png" alt="image-20211220152847894" style="zoom:50%;" />

### D.Training the Agent with Double DQN

由于普通DQN计算时会过高地估计Q值，所以采用DDQN，在原始的Double Q-Learning算法里面，有两个价值函数(value function)，一个用来选择动作（当前状态的策略），一个用来评估当前状态的价值。从而使整个强化学习更稳定。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211220171348353.png" alt="image-20211220171348353" style="zoom:50%;" />

具体思路是FL服务器首先执行随机设备选择来初始化状态，将权值状态放入DDQN中的一个DQN中，DQN生成一个动作a来为FL服务器选择设备。训练几轮之后，DQN深度学习网络的损失函数用均方误差表示：

<img src="https://www.zhihu.com/equation?tex=\ell_{t}\left(\boldsymbol{\theta}_{t}\right)=\left(Y_{t}^{\text {DoubleQ }}-Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right)\right)^{2}
" alt="\ell_{t}\left(\boldsymbol{\theta}_{t}\right)=\left(Y_{t}^{\text {DoubleQ }}-Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right)\right)^{2}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=Y_{t}^{DoubleQ}" alt="Y_{t}^{DoubleQ}" class="ee_img tr_noresize" eeimg="1"> 为：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
Y_{t}^{\text {DoubleQ }}: &=r_{t}+\gamma \max _{a} Q\left(\boldsymbol{s}_{t+1}, a ; \boldsymbol{\theta}_{t}\right) 
=r_{t}+\gamma Q\left(\boldsymbol{s}_{t}, \underset{a}{\operatorname{argmax}} Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right) ; \boldsymbol{\theta}_{t}^{\prime}\right)
\end{aligned}
" alt="\begin{aligned}
Y_{t}^{\text {DoubleQ }}: &=r_{t}+\gamma \max _{a} Q\left(\boldsymbol{s}_{t+1}, a ; \boldsymbol{\theta}_{t}\right) 
=r_{t}+\gamma Q\left(\boldsymbol{s}_{t}, \underset{a}{\operatorname{argmax}} Q\left(\boldsymbol{s}_{t}, a ; \boldsymbol{\theta}_{t}\right) ; \boldsymbol{\theta}_{t}^{\prime}\right)
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
对比DQN中的损失函数可以发现，DDQN其实就是把测试用的网络换了一个全新的，相当于用两个动作值函数Q更新 <img src="https://www.zhihu.com/equation?tex=Y_{t}^{DoubleQ}" alt="Y_{t}^{DoubleQ}" class="ee_img tr_noresize" eeimg="1"> ，使1号Q最大的动作，拿到2号Q中计算Y，其中1号的Q为预测值，2号的Q为目标值。计算完毕后二者再相减即可计算均方损失函数。

## EVALUATION

用多线程库，实现轻量级线程模拟大量设备，每个线程都运行真实的pytorch模型。我们在MNIST、FashionMNIST和CIFAR-10三个基准数据集上训练流行的CNN模型来评估FAVOR，以FedAvg和K-Center作为对比组。与FedAvg作对比，我们简单描述了方法和设置：

- **数据和模型**

选择了使得FedAvg性能最佳的超参数（CNN网络的连接、卷积层）。

- **性能指标**

将通讯的轮数作为FAVOR的性能度量。

### A.Training the DRL agent

DDQN中由2个全连接网络构成，具有512个隐藏状态，一共拥有101个权值模型，每个权值通过PCA降到100维，即输入为10100，第二层输出为100，再经过一个softmax函数。

如下图，从联合学习作业的初始化开始，当收敛到目标精度Ω时结束。MNIST的精度设为99%，FMNIST精度为85%，Cifar10精度为55%。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211220193143155.png" alt="image-20211220193143155" style="zoom:50%;" />

### B.Different Levels of Non-IID Data

本文用 <img src="https://www.zhihu.com/equation?tex=\sigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1"> 表示不同的非IID的级别， <img src="https://www.zhihu.com/equation?tex=\sigma=1" alt="\sigma=1" class="ee_img tr_noresize" eeimg="1"> 表示所有数据只属于1个标签， <img src="https://www.zhihu.com/equation?tex=\sigma=0.8" alt="\sigma=0.8" class="ee_img tr_noresize" eeimg="1"> 表示80%的数据属于一个标签，其他数据随机分布。 <img src="https://www.zhihu.com/equation?tex=\sigma=H" alt="\sigma=H" class="ee_img tr_noresize" eeimg="1"> 表示数据均匀地属于2个标签。

三大数据集训练的准确度与训练轮数的图像如图所示：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211221082918852.png" alt="image-20211221082918852" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211221082938287.png" alt="image-20211221082938287" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211221082957994.png" alt="image-20211221082957994" style="zoom:50%;" />

表中所示为CNN在MNIST上达到99%准确度时所需的通信轮数，FMNIST为85%，Cifar10位55%。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211221083536034.png" alt="image-20211221083536034" style="zoom:50%;" />

可以看到K-center并没有总是优于FedAvg。

### C.Device Selection and Weight Updates

训练 <img src="https://www.zhihu.com/equation?tex=\sigma=0.8" alt="\sigma=0.8" class="ee_img tr_noresize" eeimg="1"> ，在MNIST上训练时，通过PCA将权重降维成2维，图像可以看到，FAVOR在早期更新全局模型时，权值更新∆比FedAvg大，收敛速度更快。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/基于强化学习的非IID数据联合学习优化/image-20211221090532532.png" alt="image-20211221090532532" style="zoom:50%;" />

### D.Increasing Parallelism

结论：增加并行度（增大每一轮的选取 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 值）反而会增加训练时长。

## RELATED WORK

当前常见的优化联邦学习的方法主要就是两种：优化**通讯时长和采样效率**。
