# Multi-Level Local SGD: Distributed SGD for Heterogeneous Hierarchical Networks

[toc]

## Abstract

我们提出了多级局部SGD，一种分布式随机梯度方法，用于学习**平滑、非凸**目标。我们的网络模型由一组不相连的子网络组成，有一个中心和多个客户，每个客户效率不同。中心通过一个连接但不一定完整的网络交换信息。在我们的算法中，子网络执行一个分布式的SGD算法，使用一个中心辐射型的范例，并且中心周期性地与邻近的中心平均他们的模型。

我们首先提供了一个统一的数学框架来描述多级局部SGD算法。接着进行了理论分析，分析展示了对于收敛误差是由客户节点、中心网络拓扑、聚合次数等等因素决定的。通过仿真实验，验证了该算法在一个工作速度较慢的多层网络中的有效性。

---

## Introduction

传统分布式SGD，在每次迭代中，中心向客户发送一个模型。每个客户都对本地数据进行训练，采取一个梯度步，然后将本地训练的模型返回到中心取平均值。当中心和客户之间延迟较低时，分布式SGD很有效。延迟较高时，可以给到客户端之后执行多个梯度更新，这种方式称为**本地SGD**。

但如果客户端算力不同，则会存在掉队者。而且大部分都假设了一个中心辐射模型，这与真实情况不大相符。每个个体应该有一个集群头，然后通过一个更上层的网络通信。

基于以上原因，我们提出了多级局部学习算法MLL-SGD，每个子网络有一个中心服务器和一组客户端，上层连接一个集线器。每个子网络运行一个或多个本地SGD轮，客户端训练，然后在子网络的中心进行模型平均。集线器周期性地与集线器网络中的邻居对其模型进行平均。由于MLL-SGD平均每个本地训练周期，无论每个客户端采取多少梯度步骤，最终都不会减缓算法执行。

我们证明了MLL-SGD对于光滑和潜在的非凸损失函数的收敛性。接着假设数据是IID的，进一步分析了收敛误差与算法参数的关系。

贡献：

1、我们用异构工作者形式化了多级网络模型，并定义了该网络中训练模型的MLL-SGD算法。

2、从理论上分析了具有异质工作者的MLL-SGD的收敛保证。

3、进行了相关实验。

---

## System model and problem formulation

我们考虑有 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 个子网络， <img src="https://www.zhihu.com/equation?tex=D=\{1,...,D\}" alt="D=\{1,...,D\}" class="ee_img tr_noresize" eeimg="1"> ，每一个子网络 <img src="https://www.zhihu.com/equation?tex=d\in D" alt="d\in D" class="ee_img tr_noresize" eeimg="1"> 都有一个中心和一组客户端 <img src="https://www.zhihu.com/equation?tex=M^{(d)}" alt="M^{(d)}" class="ee_img tr_noresize" eeimg="1"> ，并且数量上 <img src="https://www.zhihu.com/equation?tex=|M^{(d)}|=N^{(d)}" alt="|M^{(d)}|=N^{(d)}" class="ee_img tr_noresize" eeimg="1"> ，我们将客户端所有的集合定义为 <img src="https://www.zhihu.com/equation?tex=M=\bigcup^D_{d=1}M^{(d)}" alt="M=\bigcup^D_{d=1}M^{(d)}" class="ee_img tr_noresize" eeimg="1"> ，令 <img src="https://www.zhihu.com/equation?tex=|M|=N" alt="|M|=N" class="ee_img tr_noresize" eeimg="1"> ，每一个客户端 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 都有一组训练集 <img src="https://www.zhihu.com/equation?tex=S^{(i)}" alt="S^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，令 <img src="https://www.zhihu.com/equation?tex=S=\bigcup^N_{i=1}S^{(i)}" alt="S=\bigcup^N_{i=1}S^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，所有 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 集线器的集合用 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 表示。集线器通过一个 <img src="https://www.zhihu.com/equation?tex=G=(C,E)" alt="G=(C,E)" class="ee_img tr_noresize" eeimg="1"> 进行通信，令 <img src="https://www.zhihu.com/equation?tex=N_d=\{j|e_{d,j}\in E\}" alt="N_d=\{j|e_{d,j}\in E\}" class="ee_img tr_noresize" eeimg="1"> 表示在网络 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 的中心的邻居节点的集合。

设模型参数是 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> ，我们的目的是使损失函数的均值最小化，即：

<img src="https://www.zhihu.com/equation?tex=F(\boldsymbol{x})=\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} f(\boldsymbol{x} ; s)
" alt="F(\boldsymbol{x})=\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} f(\boldsymbol{x} ; s)
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=f(\cdot)" alt="f(\cdot)" class="ee_img tr_noresize" eeimg="1"> 表示损失函数。客户端每一次局部迭代在本地运行SGD时，会抽出一小批数据执行，设 <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1"> 是随机抽样的一小批数据，并且令 <img src="https://www.zhihu.com/equation?tex=g(\boldsymbol{x} ; \xi)=\frac{1}{|\xi|} \sum_{s \in \xi} \nabla f(\boldsymbol{x} ; s)" alt="g(\boldsymbol{x} ; \xi)=\frac{1}{|\xi|} \sum_{s \in \xi} \nabla f(\boldsymbol{x} ; s)" class="ee_img tr_noresize" eeimg="1"> 为这一小批数据的梯度，简写为 <img src="https://www.zhihu.com/equation?tex=g(x)" alt="g(x)" class="ee_img tr_noresize" eeimg="1"> 。

**假设1：**

目标函数和小批量梯度满足以下条件：

1、目标函数 <img src="https://www.zhihu.com/equation?tex=F" alt="F" class="ee_img tr_noresize" eeimg="1"> 是连续可微的，并且梯度满足： <img src="https://www.zhihu.com/equation?tex=\|\nabla F(\boldsymbol{x})-\nabla F(\boldsymbol{y})\|_{2} \leq L\|\boldsymbol{x}-\boldsymbol{y}\|_{2}" alt="\|\nabla F(\boldsymbol{x})-\nabla F(\boldsymbol{y})\|_{2} \leq L\|\boldsymbol{x}-\boldsymbol{y}\|_{2}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=L>0" alt="L>0" class="ee_img tr_noresize" eeimg="1"> 。

2、 <img src="https://www.zhihu.com/equation?tex=F" alt="F" class="ee_img tr_noresize" eeimg="1"> 有下界。 <img src="https://www.zhihu.com/equation?tex=F(x)\geq F_{inf}>-\infty" alt="F(x)\geq F_{inf}>-\infty" class="ee_img tr_noresize" eeimg="1"> 。

3、小批量梯度是没有偏差的，即 <img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\xi \mid \boldsymbol{x}}[g(\boldsymbol{x})]=\nabla F(\boldsymbol{x})" alt="\mathbb{E}_{\xi \mid \boldsymbol{x}}[g(\boldsymbol{x})]=\nabla F(\boldsymbol{x})" class="ee_img tr_noresize" eeimg="1"> 。

4、存在标量 <img src="https://www.zhihu.com/equation?tex=\beta \geq0" alt="\beta \geq0" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\sigma \geq0" alt="\sigma \geq0" class="ee_img tr_noresize" eeimg="1"> ，使得 <img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\xi \mid \boldsymbol{x}}\|g(\boldsymbol{x})-\nabla F(\boldsymbol{x})\|_{2}^{2} \leq \beta\|\nabla F(\boldsymbol{x})\|_{2}^{2}+\sigma^{2}" alt="\mathbb{E}_{\xi \mid \boldsymbol{x}}\|g(\boldsymbol{x})-\nabla F(\boldsymbol{x})\|_{2}^{2} \leq \beta\|\nabla F(\boldsymbol{x})\|_{2}^{2}+\sigma^{2}" class="ee_img tr_noresize" eeimg="1"> 。

上述假设1说明梯度不会变化太快，2说明了梯度的下界，3和4说明了每个客户端的数据**可以作为整体数据的具有相同方差的无偏差估计**。

---

## Algorithm

下面介绍多级局部SGD算法。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220121145300060.png" alt="image-20220121145300060" style="zoom:50%;" />

每个子网络并行排列，并且中心定期对模型进行平均。localSGD的步骤对应第5-10行，每个中心和客户端都会存储一个模型副本。对于客户端 <img src="https://www.zhihu.com/equation?tex=i\in M^{(d)}" alt="i\in M^{(d)}" class="ee_img tr_noresize" eeimg="1"> ，其局部模型为 <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，我们表示中心 <img src="https://www.zhihu.com/equation?tex=d" alt="d" class="ee_img tr_noresize" eeimg="1"> 处的模型为 <img src="https://www.zhihu.com/equation?tex=y^{(d)}" alt="y^{(d)}" class="ee_img tr_noresize" eeimg="1"> ，客户端进行多次迭代，如第7行。为了表示每个工人的不同计算速率，我们使用了一种概率方法，每 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 个时间步里，每个客户端 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 执行 <img src="https://www.zhihu.com/equation?tex=\gamma^{(i)}" alt="\gamma^{(i)}" class="ee_img tr_noresize" eeimg="1"> 次迭代，其中 <img src="https://www.zhihu.com/equation?tex=\gamma^{(i)}<\gamma" alt="\gamma^{(i)}<\gamma" class="ee_img tr_noresize" eeimg="1"> 。因此我们定义一个 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 维的向量 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=p_i=\frac{\gamma^{(i)}}{\gamma}" alt="p_i=\frac{\gamma^{(i)}}{\gamma}" class="ee_img tr_noresize" eeimg="1"> 是在第 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 次迭代中的梯度步长概率，客户端 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 在第 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 次的模型更新为：

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{x}_{k+1}^{(i)}=\boldsymbol{x}_{k}^{(i)}-\eta \boldsymbol{g}_{k}^{(i)}
" alt="\boldsymbol{x}_{k+1}^{(i)}=\boldsymbol{x}_{k}^{(i)}-\eta \boldsymbol{g}_{k}^{(i)}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> 为学习率， <img src="https://www.zhihu.com/equation?tex=g_k^{(i)}" alt="g_k^{(i)}" class="ee_img tr_noresize" eeimg="1"> 为：

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{g}_{k}^{(i)}= \begin{cases}g\left(\boldsymbol{x}_{k}^{(i)}\right) & \text { w/ probability } \boldsymbol{p}_{i} \\ \mathbf{0} & \text { w/ probability } 1-\boldsymbol{p}_{i}\end{cases}
" alt="\boldsymbol{g}_{k}^{(i)}= \begin{cases}g\left(\boldsymbol{x}_{k}^{(i)}\right) & \text { w/ probability } \boldsymbol{p}_{i} \\ \mathbf{0} & \text { w/ probability } 1-\boldsymbol{p}_{i}\end{cases}
" class="ee_img tr_noresize" eeimg="1">
对于每一个客户端 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> ，我们分配一个正权重 <img src="https://www.zhihu.com/equation?tex=w^{(i)}" alt="w^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，设 <img src="https://www.zhihu.com/equation?tex=v^{(i)}" alt="v^{(i)}" class="ee_img tr_noresize" eeimg="1"> 是一个归一化权重，其表示为：

<img src="https://www.zhihu.com/equation?tex=v^{(i)}=\frac{w^{(i)}}{\sum_{j \in \mathcal{M}^{(d(i))} w^{(j)}}}
" alt="v^{(i)}=\frac{w^{(i)}}{\sum_{j \in \mathcal{M}^{(d(i))} w^{(j)}}}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=d(i)" alt="d(i)" class="ee_img tr_noresize" eeimg="1"> 是客户端 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 所属的网络。每一个中心都将其模型更新为其子网络中工人模型的加权平均： <img src="https://www.zhihu.com/equation?tex=\boldsymbol{y}^{(d)}=\sum_{i \in \mathcal{M}^{(d)}} v^{(i)} \boldsymbol{x}^{(i)}" alt="\boldsymbol{y}^{(d)}=\sum_{i \in \mathcal{M}^{(d)}} v^{(i)} \boldsymbol{x}^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，如果所有的客户端都被平等对待，则 <img src="https://www.zhihu.com/equation?tex=w^{(i)}=1" alt="w^{(i)}=1" class="ee_img tr_noresize" eeimg="1"> ，一般情况下令 <img src="https://www.zhihu.com/equation?tex=w^{(i)}=|S^{(i)}|" alt="w^{(i)}=|S^{(i)}|" class="ee_img tr_noresize" eeimg="1"> （类似于FedAvg）。

当每一个子网络进行 <img src="https://www.zhihu.com/equation?tex=q\cdot \gamma" alt="q\cdot \gamma" class="ee_img tr_noresize" eeimg="1"> 次迭代后，各中心和其邻居进行通信并平均模型（算法12行），分配给每个中心模型的权重由一个 <img src="https://www.zhihu.com/equation?tex=D\times D" alt="D\times D" class="ee_img tr_noresize" eeimg="1"> 的矩阵 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 来决定：

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{y}^{(d)}=\sum_{j \in \mathcal{N}^{(d)}} \mathbf{H}_{j, d} \boldsymbol{y}^{(j)}
" alt="\boldsymbol{y}^{(d)}=\sum_{j \in \mathcal{N}^{(d)}} \mathbf{H}_{j, d} \boldsymbol{y}^{(j)}
" class="ee_img tr_noresize" eeimg="1">
定义网络中的总权重为： <img src="https://www.zhihu.com/equation?tex=w_{t o t}=\sum_{i \in \mathcal{M}} w^{(i)}" alt="w_{t o t}=\sum_{i \in \mathcal{M}} w^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，令 <img src="https://www.zhihu.com/equation?tex=b" alt="b" class="ee_img tr_noresize" eeimg="1"> 是一个 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 维的向量，并且 <img src="https://www.zhihu.com/equation?tex=b_d=(\sum_{i\in M^{(d)}}w^{(i)})/w_{tot}" alt="b_d=(\sum_{i\in M^{(d)}}w^{(i)})/w_{tot}" class="ee_img tr_noresize" eeimg="1"> 。则我们假设矩阵 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 满足以下几种条件：

**假设2：**

1、如果 <img src="https://www.zhihu.com/equation?tex=(i,j)\in E" alt="(i,j)\in E" class="ee_img tr_noresize" eeimg="1"> ，那么 <img src="https://www.zhihu.com/equation?tex=H_{i,j}>0" alt="H_{i,j}>0" class="ee_img tr_noresize" eeimg="1"> ，否则 <img src="https://www.zhihu.com/equation?tex=H_{i,j}=0" alt="H_{i,j}=0" class="ee_img tr_noresize" eeimg="1"> 。

2、 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 为列随机，即 <img src="https://www.zhihu.com/equation?tex=\sum_{i=1}^{D} \mathbf{H}_{i, j}=1" alt="\sum_{i=1}^{D} \mathbf{H}_{i, j}=1" class="ee_img tr_noresize" eeimg="1"> 。

3、对于所有的 <img src="https://www.zhihu.com/equation?tex=i,j\in D" alt="i,j\in D" class="ee_img tr_noresize" eeimg="1"> ，我们都有 <img src="https://www.zhihu.com/equation?tex=b_iH_{i,j}=b_jH_{j,i}" alt="b_iH_{i,j}=b_jH_{j,i}" class="ee_img tr_noresize" eeimg="1"> 。

上述假设说明 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 有一个简单的特征向量，右特征向量为 <img src="https://www.zhihu.com/equation?tex=b" alt="b" class="ee_img tr_noresize" eeimg="1"> ，左特征向量均为1，除此以外没有其他特征向量。通过以上方式定义 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> ，客户端梯度的贡献和其权重成比例的结合在了一起，从而更好地拓展到多级网络模型。

---

## Analysis

可以看到，中心的模型本质上是无状态的，因为当中心的模型平均之后，其被分配到每个客户端。因此我们的分析集中在客户端是怎样进化的。我们首先根据客户端模型的演化提出了一个等效的MLL-SGD算法的公式，然后给出了我们关于MLL-SGD收敛的主要结果。

系统的行为可以用以下客户端模型更新规则作为总结：

<img src="https://www.zhihu.com/equation?tex=\mathbf{X}_{k+1}=\left(\mathbf{X}_{k}-\eta \mathbf{G}_{k}\right) \mathbf{T}_{k}
" alt="\mathbf{X}_{k+1}=\left(\mathbf{X}_{k}-\eta \mathbf{G}_{k}\right) \mathbf{T}_{k}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=n \times N" alt="n \times N" class="ee_img tr_noresize" eeimg="1"> 的矩阵 <img src="https://www.zhihu.com/equation?tex=\mathbf{X}_{k}=\left[\boldsymbol{x}_{k}^{(1)}, \ldots, \boldsymbol{x}_{k}^{(N)}\right]" alt="\mathbf{X}_{k}=\left[\boldsymbol{x}_{k}^{(1)}, \ldots, \boldsymbol{x}_{k}^{(N)}\right]" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=n\times N" alt="n\times N" class="ee_img tr_noresize" eeimg="1"> 的矩阵 <img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{k}=\left[\boldsymbol{g}_{k}^{(1)}, \ldots, \boldsymbol{g}_{k}^{(N)}\right]" alt="\mathbf{G}_{k}=\left[\boldsymbol{g}_{k}^{(1)}, \ldots, \boldsymbol{g}_{k}^{(N)}\right]" class="ee_img tr_noresize" eeimg="1"> ，还有 <img src="https://www.zhihu.com/equation?tex=N\times N" alt="N\times N" class="ee_img tr_noresize" eeimg="1"> 的矩阵是一个时变算子，它捕获了MLL-SGD中的三个阶段：局部迭代，每个子网络中的中心平均，和所有中心的平均，定义 <img src="https://www.zhihu.com/equation?tex=T_k" alt="T_k" class="ee_img tr_noresize" eeimg="1"> 为：

<img src="https://www.zhihu.com/equation?tex=\mathbf{T}_{k}= \begin{cases}\mathbf{Z} & \text { if } k \bmod q \tau=0 \\ \mathbf{V} & \text { if } k \bmod \tau=0 \text { and } k \bmod q \tau \neq 0 \\ \mathbf{I} & \text { otherwise }\end{cases}
" alt="\mathbf{T}_{k}= \begin{cases}\mathbf{Z} & \text { if } k \bmod q \tau=0 \\ \mathbf{V} & \text { if } k \bmod \tau=0 \text { and } k \bmod q \tau \neq 0 \\ \mathbf{I} & \text { otherwise }\end{cases}
" class="ee_img tr_noresize" eeimg="1">
对于本地的local计算， <img src="https://www.zhihu.com/equation?tex=T_k=I" alt="T_k=I" class="ee_img tr_noresize" eeimg="1"> ，对于子网络的平均， <img src="https://www.zhihu.com/equation?tex=V" alt="V" class="ee_img tr_noresize" eeimg="1"> 是一个 <img src="https://www.zhihu.com/equation?tex=N\times N" alt="N\times N" class="ee_img tr_noresize" eeimg="1"> 的一个分块对角矩阵，每一块的矩阵 <img src="https://www.zhihu.com/equation?tex=V^{(d)}" alt="V^{(d)}" class="ee_img tr_noresize" eeimg="1"> 表示为 <img src="https://www.zhihu.com/equation?tex=\mathbf{V}_{i, j}^{(d)}=v^{(i)}" alt="\mathbf{V}_{i, j}^{(d)}=v^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，最后，所有子网络聚合的矩阵 <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1"> 表示为：

<img src="https://www.zhihu.com/equation?tex=\mathbf{Z}_{i, j}=\mathbf{H}_{d(i), d(j)} v^{(i)}
" alt="\mathbf{Z}_{i, j}=\mathbf{H}_{d(i), d(j)} v^{(i)}
" class="ee_img tr_noresize" eeimg="1">
设 <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1"> 为一个 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 维向量， <img src="https://www.zhihu.com/equation?tex=\boldsymbol{a}_{i}=\frac{w^{(i)}}{w_{t o t}}" alt="\boldsymbol{a}_{i}=\frac{w^{(i)}}{w_{t o t}}" class="ee_img tr_noresize" eeimg="1"> ，表示客户 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的权重，这些属性是必要的(但不是充分的)，以确保工人模型收敛到一个共识模型，其中每个工人的更新已根据工人的权重合并。

我们定义了一个加权平均模型:

<img src="https://www.zhihu.com/equation?tex=\boldsymbol{u}_{k}=\mathbf{X}_{k} \boldsymbol{a}
" alt="\boldsymbol{u}_{k}=\mathbf{X}_{k} \boldsymbol{a}
" class="ee_img tr_noresize" eeimg="1">
通过对上式左右两边同时乘以一个a，可以得到：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227111926155.png" alt="image-20220227111926155" style="zoom:50%;" />

我们注意到 <img src="https://www.zhihu.com/equation?tex=u_k" alt="u_k" class="ee_img tr_noresize" eeimg="1"> 是通过使用几个小批量梯度的加权平均的随机梯度下降步来更新的。由于 <img src="https://www.zhihu.com/equation?tex=F(·)" alt="F(·)" class="ee_img tr_noresize" eeimg="1"> 可以是非凸的，所以SGD可以收敛到局部最小值或鞍点。因此，我们研究了当 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 增加时 <img src="https://www.zhihu.com/equation?tex=u_k" alt="u_k" class="ee_img tr_noresize" eeimg="1"> 的梯度。

**定理1：**

在假设1和假设2下，如果 <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> 对于所有的 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 都满足：

<img src="https://www.zhihu.com/equation?tex=\left(4 \boldsymbol{p}_{i}-\boldsymbol{p}_{i}^{2}-2\right) \geq \eta L\left(\boldsymbol{a}_{i} \boldsymbol{p}_{i}(\beta+1)-\boldsymbol{a}_{i} \boldsymbol{p}_{i}^{2}+\boldsymbol{p}_{i}^{2}\right)+8 L^{2} \eta^{2} q^{2} \tau^{2} \Gamma
" alt="\left(4 \boldsymbol{p}_{i}-\boldsymbol{p}_{i}^{2}-2\right) \geq \eta L\left(\boldsymbol{a}_{i} \boldsymbol{p}_{i}(\beta+1)-\boldsymbol{a}_{i} \boldsymbol{p}_{i}^{2}+\boldsymbol{p}_{i}^{2}\right)+8 L^{2} \eta^{2} q^{2} \tau^{2} \Gamma
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\Gamma=\frac{\zeta}{1-\zeta^{2}}+\frac{2}{1-\zeta}+\frac{\zeta}{(1-\zeta)^{2}}" alt="\Gamma=\frac{\zeta}{1-\zeta^{2}}+\frac{2}{1-\zeta}+\frac{\zeta}{(1-\zeta)^{2}}" class="ee_img tr_noresize" eeimg="1"> 并且 <img src="https://www.zhihu.com/equation?tex=\zeta=\max \left\{\left|\lambda_{2}(\mathbf{H})\right|,\left|\lambda_{N}(\mathbf{H})\right|\right\}" alt="\zeta=\max \left\{\left|\lambda_{2}(\mathbf{H})\right|,\left|\lambda_{N}(\mathbf{H})\right|\right\}" class="ee_img tr_noresize" eeimg="1"> 。

则经过 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 次迭代平均的平均**模型梯度的期望平方范数**有界如下:

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227114241574.png" alt="image-20220227114241574" style="zoom:50%;" />

式子13的第一部分和普通中心SGD一样，当 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 趋于无穷时，这一部分为0。第二个阶段也类似于中央SGD。如果随机梯度的方差较大，则收敛误差较大。

第三和第四项是依赖于中心网络拓扑结构的可加性误差。 <img src="https://www.zhihu.com/equation?tex=\zeta" alt="\zeta" class="ee_img tr_noresize" eeimg="1"> 的值由H的第二大特征值(大小)给出，表明了枢纽网络的稀疏性。当客户端权重一致时，一个完全连通的枢纽图 <img src="https://www.zhihu.com/equation?tex=G" alt="G" class="ee_img tr_noresize" eeimg="1"> 将 <img src="https://www.zhihu.com/equation?tex=\zeta =0" alt="\zeta =0" class="ee_img tr_noresize" eeimg="1"> ，而一个稀疏的 <img src="https://www.zhihu.com/equation?tex=G" alt="G" class="ee_img tr_noresize" eeimg="1"> 通常将 <img src="https://www.zhihu.com/equation?tex=\zeta" alt="\zeta" class="ee_img tr_noresize" eeimg="1"> 接近1。有趣的是， <img src="https://www.zhihu.com/equation?tex=\zeta" alt="\zeta" class="ee_img tr_noresize" eeimg="1"> 只依赖于 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> ，而不是 <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=V" alt="V" class="ee_img tr_noresize" eeimg="1"> ，这意味着收敛误差不依赖于客户端权重在子网络中的分布。

我们还注意到第三和第四项取决于工人的加权平均概率 <img src="https://www.zhihu.com/equation?tex=P" alt="P" class="ee_img tr_noresize" eeimg="1"> 。收敛误差随着平均工人操作率的增加而增加。有趣的是，收敛误差并不依赖于 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 的分布，这意味着具有相同平均概率的倾斜均匀分布将具有相同的收敛误差。我们观察到，在给定某些概率的情况下， <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> 的条件并不总是满足。

第三项和第四项也随着 <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> (每个枢纽网络平均步数和子网络内部平均步的局部迭代次数)的增大而增大。工人在当地培训的时间越长，而不协调他们的模型，他们的模型就会发散越多，导致更大的收敛误差。 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 的影响比 <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> 更大，**所以应多进行内部的迭代计算。**

在下面的推论中，我们分析了算法1的收敛速度：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227152500606.png" alt="image-20220227152500606" style="zoom:50%;" />

MLL-SGD与Local SGD和HL-SGD具有相同的渐近收敛速度。

---

## Experiments

数据：MNIST和Cifar10

对比实验：

- 分布式SGD：相当于Fedavg， <img src="https://www.zhihu.com/equation?tex=q=\gamma=1" alt="q=\gamma=1" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=p_i=1" alt="p_i=1" class="ee_img tr_noresize" eeimg="1"> ，作为一个标准测试
- localSGD：集线器网络全连接，当对于所有的 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 都有 <img src="https://www.zhihu.com/equation?tex=a_i=1/N" alt="a_i=1/N" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=p_i=1" alt="p_i=1" class="ee_img tr_noresize" eeimg="1"> 的时候，localSGD和MLLSGD相同并且 <img src="https://www.zhihu.com/equation?tex=q=1" alt="q=1" class="ee_img tr_noresize" eeimg="1"> 。
- HL-SGD：拓展为 <img src="https://www.zhihu.com/equation?tex=q>1" alt="q>1" class="ee_img tr_noresize" eeimg="1"> 。

对于所有的实验，我们让localSGD局部 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 为32，对于HL-SGD和MLL-SGD令 <img src="https://www.zhihu.com/equation?tex=q\gamma=32" alt="q\gamma=32" class="ee_img tr_noresize" eeimg="1"> ，每32次计算进行一次统计。

我们搞了10个中心，每个中心连接10个客户端。对于MLL-SGD设置了2种构型，分别是 <img src="https://www.zhihu.com/equation?tex=\gamma=4,q=8" alt="\gamma=4,q=8" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\gamma=8,q=4" alt="\gamma=8,q=4" class="ee_img tr_noresize" eeimg="1"> 。对于分布式SGD和localSGD，集线器相当于连接通信，客户端分为5组，每组20人，每个组被分到的数据集百分比：5%、10%、20%、25%、40%，权重根据数据集大小进行分配。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227160413101.png" alt="image-20220227160413101" style="zoom:50%;" />

图ac是损失，bd是精确率，随着 <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1"> 的增加，在保持 <img src="https://www.zhihu.com/equation?tex=q\gamma = 32" alt="q\gamma = 32" class="ee_img tr_noresize" eeimg="1"> 的情况下，MLL-SGD改善并接近分布式SGD。因此，增加子网络训练轮数可以改善MLL-SGD的收敛行为。

我们研究了子网络的数量和规模如何影响MLL-SGD的收敛，从100个工人中，我们将他们分配到5个、10个和20个子网络中。并且在保持网络连接的同时产生最大的 <img src="https://www.zhihu.com/equation?tex=\zeta" alt="\zeta" class="ee_img tr_noresize" eeimg="1"> 。搭配上1个中心和100个工人的localSGD测试，得到结果：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227162150458.png" alt="image-20220227162150458" style="zoom:50%;" />

在CNN的情况下，在MLL-SGD变化中，训练损失的差异是最小的。在ResNet的情况下，随着枢纽数量的增加，收敛速度下降。这与定理1是一致的，因为增加的数集线器对应增加的 <img src="https://www.zhihu.com/equation?tex=\zeta" alt="\zeta" class="ee_img tr_noresize" eeimg="1"> 。

接下来是不同分布的影响，我们比较了四种不同的MLL-SGD设置，所有这些都包括一个完整的集线器网络，10个集线器，每个有10个工人， <img src="https://www.zhihu.com/equation?tex=a_i=1/N" alt="a_i=1/N" class="ee_img tr_noresize" eeimg="1"> ，工人之间的平均概率为0.55：（1）每个worker有固定的概率0.55。（2）每个子网络中，工人的概率从0.1-1均匀分布。（3）90个工人0.5和10个工人的1（倾斜1）。（4） 90个工人0.6和10个工人0.1（倾斜2）。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227163915437.png" alt="image-20220227163915437" style="zoom:50%;" />

可以看到准确率相同，因为概率相同，和之前的推导相同。

最后比较了MLL-SGD与其他几种较慢的算法：localSGD和HL-SGD，在每一个时间段中，每个工人采取概率 <img src="https://www.zhihu.com/equation?tex=p_i" alt="p_i" class="ee_img tr_noresize" eeimg="1"> 进行梯度更新，如果 <img src="https://www.zhihu.com/equation?tex=p_i=1" alt="p_i=1" class="ee_img tr_noresize" eeimg="1"> 则代表有多少个时间段就会有多少次梯度更新。MLL-SGD在 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 个时间步后直接聚合，但其他算法需要等待工人们都运行了 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 个时间步。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/多层次SGD/image-20220227165817535.png" alt="image-20220227165817535" style="zoom:50%;" />

在此实验设置中，等待慢速工作者不利于整体收敛时间。
