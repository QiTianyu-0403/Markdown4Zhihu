# FedMes: Speeding Up Federated Learning With Multiple Edge Servers

我们的核心思想是利用位于相邻边缘服务器(ESs)重叠覆盖区域的客户端，在模型下载阶段，重叠区域的客户端收到来自不同ESs的多个模型，对收到的模型取平均，然后用各自的本地数据更新平均模型。

这些客户端通过广播将更新后的模型发送给多个ESs，作为服务器之间**共享训练过的模型的桥梁**。即使某些ESs在其覆盖区域内给定有偏差的数据集，其训练过程也可以通过重叠区域内的客户端得到**相邻服务器**的帮助。

## Introduction

主要在以下部分实现：1、通信瓶颈；2、noniid数据处理

当下问题：云和客户端相距太远，上传下载数据延迟过大，本文中，考虑到FL对延迟敏感的应用或紧急事件，ps的位置位于边缘。

以上问题仍存在：在实际系统中，边缘服务器的覆盖通常是有限的(例如，无线蜂窝网络)；在边缘服务器的覆盖范围内，没有足够数量的客户端来训练具有足够精度的全局模型，因此，单一边缘服务器的有限覆盖可能包括有偏差的数据集，从而可能导致训练后的有偏差模型。

- 主要贡献

提出了FedMes算法，与传统算法相比，其不需要与云有昂贵的通信，我们的关键理念是利用位于ESs覆盖之间重叠区域的客户。

在模型下载阶段，每个ES将当前模型发送给其覆盖区域内的客户端，重叠区域中的这些客户端取接收到的模型的平均值，然后根据其本地数据更新模型。然后，每个客户端将其更新后的模型发送到相应的ES或ESs，这些ES或ESs在每个ES上聚合。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220302091547435.png" alt="image-20220302091547435" style="zoom:50%;" />

设客户 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 位于 <img src="https://www.zhihu.com/equation?tex=ES_i" alt="ES_i" class="ee_img tr_noresize" eeimg="1"> 的非重叠区域，客户 <img src="https://www.zhihu.com/equation?tex=l" alt="l" class="ee_img tr_noresize" eeimg="1"> 位于 <img src="https://www.zhihu.com/equation?tex=ES_i" alt="ES_i" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=ES_j" alt="ES_j" class="ee_img tr_noresize" eeimg="1"> 之间的重叠区域，所以客户 <img src="https://www.zhihu.com/equation?tex=l" alt="l" class="ee_img tr_noresize" eeimg="1"> 可以作为一个桥梁共享2个 <img src="https://www.zhihu.com/equation?tex=ES" alt="ES" class="ee_img tr_noresize" eeimg="1"> 之间的模型，从这个角度来看,即使一些训练样本仅在特定ES的覆盖范围内，这些数据仍然可以辅助其他服务器的训练过程。因此，与基于云的FL系统相比，本文提出的方案**不需要与中央云服务器(位于ESs的较高层)进行昂贵的通信**以实现模型同步

主要贡献：

1、提出了FedMes

2、推导了FedMes的收敛界限

3、通过实验，对比中心云服务器的通信和不考虑ES之间的重叠，FedMes性能显著提升。

- 相关工作

1、之前的工作共同想法是执行分层FL，但仍会有很大的开销。

2、另一项工作是去中心化的FL，客户端可以直接和其邻居交换信息。可以达到较好的收敛速度(即保证收敛)，但由于每个客户端需要与大量的邻居进行模型交换，因此每轮的时间延迟较大。当网络连接稀疏时，每一轮的时延变小，但收敛性不保证，特别是在非iid数据分布情况下。

## System Model

- **联邦学习背景**

设 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 为客户端数量， <img src="https://www.zhihu.com/equation?tex=n_k" alt="n_k" class="ee_img tr_noresize" eeimg="1"> 为客户端 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 的数据量， <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 为数据量总量，在客户端 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 中的第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 个数据样本表示为 <img src="https://www.zhihu.com/equation?tex=x_{k,i}" alt="x_{k,i}" class="ee_img tr_noresize" eeimg="1"> ，我们的目标是优化：

<img src="https://www.zhihu.com/equation?tex=\min _{\mathbf{w}} F(\mathbf{w})=\min _{\mathbf{w}} \sum_{k=1}^{K} \frac{n_{k}}{n} F_{k}(\mathbf{w})
" alt="\min _{\mathbf{w}} F(\mathbf{w})=\min _{\mathbf{w}} \sum_{k=1}^{K} \frac{n_{k}}{n} F_{k}(\mathbf{w})
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=F_k(\mathbf{w})" alt="F_k(\mathbf{w})" class="ee_img tr_noresize" eeimg="1"> 是客户端 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 的损失函数，可以写为 <img src="https://www.zhihu.com/equation?tex=F_{k}(\mathbf{w})=\frac{1}{n_{k}} \sum_{i=1}^{n_{k}} \ell\left(x_{k, i} ; \mathbf{w}\right)" alt="F_{k}(\mathbf{w})=\frac{1}{n_{k}} \sum_{i=1}^{n_{k}} \ell\left(x_{k, i} ; \mathbf{w}\right)" class="ee_img tr_noresize" eeimg="1"> 。

对于传统的FedAvg，Ps提供模型 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}(t)" alt="\mathbf{w}(t)" class="ee_img tr_noresize" eeimg="1"> ，训练 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 个轮次：

<img src="https://www.zhihu.com/equation?tex=\mathbf{w}_{k}(t+i+1)=\mathbf{w}_{k}(t+i)-\eta_{t+i} \nabla F_{k}\left(\mathbf{w}_{k}(t+i), \xi_{k}(t+i)\right)
" alt="\mathbf{w}_{k}(t+i+1)=\mathbf{w}_{k}(t+i)-\eta_{t+i} \nabla F_{k}\left(\mathbf{w}_{k}(t+i), \xi_{k}(t+i)\right)
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> 是学习率， <img src="https://www.zhihu.com/equation?tex=\xi_k" alt="\xi_k" class="ee_img tr_noresize" eeimg="1"> 是从客户 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 随机选择的一组样本。PS再从 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 个客户端中选取一个集合 <img src="https://www.zhihu.com/equation?tex=S_{t+E}" alt="S_{t+E}" class="ee_img tr_noresize" eeimg="1"> 进行聚合：

<img src="https://www.zhihu.com/equation?tex=\mathbf{w}(t+E)=\sum_{k \in S_{t+E}} \frac{n_{k}}{\sum_{k \in S_{t+E}} n_{k}} \mathbf{w}_{k}(t+E)
" alt="\mathbf{w}(t+E)=\sum_{k \in S_{t+E}} \frac{n_{k}}{\sum_{k \in S_{t+E}} n_{k}} \mathbf{w}_{k}(t+E)
" class="ee_img tr_noresize" eeimg="1">

- **问题设置：多个ES重叠**

我们将一个ES覆盖的区域称为单元，设一共有 <img src="https://www.zhihu.com/equation?tex=L" alt="L" class="ee_img tr_noresize" eeimg="1"> 个单元，一个客户端有可能包含在多个单元内。

设 <img src="https://www.zhihu.com/equation?tex=C_i" alt="C_i" class="ee_img tr_noresize" eeimg="1"> 是在单元 <img src="https://www.zhihu.com/equation?tex=i\in\{1,2,...,L\}" alt="i\in\{1,2,...,L\}" class="ee_img tr_noresize" eeimg="1"> 中的客户索引集，设 <img src="https://www.zhihu.com/equation?tex=U_i" alt="U_i" class="ee_img tr_noresize" eeimg="1"> 是 <img src="https://www.zhihu.com/equation?tex=C_i" alt="C_i" class="ee_img tr_noresize" eeimg="1"> 的子集表示非重叠用户， <img src="https://www.zhihu.com/equation?tex=V_{i,j}" alt="V_{i,j}" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=C_i" alt="C_i" class="ee_img tr_noresize" eeimg="1"> 的子集表示重叠部分的客户。在此只考虑了2个单元重叠的情况，则 <img src="https://www.zhihu.com/equation?tex=C_i" alt="C_i" class="ee_img tr_noresize" eeimg="1"> 可以表示为：

<img src="https://www.zhihu.com/equation?tex=C_{i}=U_{i} \cup\left(\underset{j \in[L] \backslash\{i\}}{\cup} V_{i, j}\right)
" alt="C_{i}=U_{i} \cup\left(\underset{j \in[L] \backslash\{i\}}{\cup} V_{i, j}\right)
" class="ee_img tr_noresize" eeimg="1">
对于所有的 <img src="https://www.zhihu.com/equation?tex=i\in\{1,2,...,L\}" alt="i\in\{1,2,...,L\}" class="ee_img tr_noresize" eeimg="1"> 系统覆盖范围可以表示为：

<img src="https://www.zhihu.com/equation?tex=C=\{1,2,...,K\}=\cup_{i=1}^LU_i\cup(\cup_{i=1}^L\cup_{j=i+1}^L)V_{i,j}
" alt="C=\{1,2,...,K\}=\cup_{i=1}^LU_i\cup(\cup_{i=1}^L\cup_{j=i+1}^L)V_{i,j}
" class="ee_img tr_noresize" eeimg="1">

## FedMes Algorithm

- **算法描述**

每个客户端从相应的 <img src="https://www.zhihu.com/equation?tex=ES" alt="ES" class="ee_img tr_noresize" eeimg="1"> 中下载模型，如果客户端位于非重叠部分，则 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}_k(t)=\mathbf{w}^{(i)}(t)" alt="\mathbf{w}_k(t)=\mathbf{w}^{(i)}(t)" class="ee_img tr_noresize" eeimg="1"> ，对于 <img src="https://www.zhihu.com/equation?tex=k\in V_{i,j}" alt="k\in V_{i,j}" class="ee_img tr_noresize" eeimg="1"> 的客户端模型为：

<img src="https://www.zhihu.com/equation?tex=\begin{gathered}
\mathbf{w}_{k}(t)=\frac{1}{\sum_{k \in S_{t}^{(i)}} n_{k}+\sum_{k \in S_{t}^{(j)} n_{k}}} 
\times\left(\sum_{k \in S_{t}^{(i)}} n_{k} \mathbf{w}^{(i)}(t)+\sum_{k \in S_{t}^{(j)}} n_{k} \mathbf{w}^{(j)}(t)\right)
\end{gathered}
" alt="\begin{gathered}
\mathbf{w}_{k}(t)=\frac{1}{\sum_{k \in S_{t}^{(i)}} n_{k}+\sum_{k \in S_{t}^{(j)} n_{k}}} 
\times\left(\sum_{k \in S_{t}^{(i)}} n_{k} \mathbf{w}^{(i)}(t)+\sum_{k \in S_{t}^{(j)}} n_{k} \mathbf{w}^{(j)}(t)\right)
\end{gathered}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=S_t^{(i)}" alt="S_t^{(i)}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=S_t^{(j)}" alt="S_t^{(j)}" class="ee_img tr_noresize" eeimg="1"> 分别是上一个步骤中将结果发送给ES <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 的客户端集合。通过此方法，对**在前面的聚合步骤中使用更多训练样本的ES的聚合模型给予更大的权重**。如果两个ES权重（训练数据量）一样，则可以表示为：

<img src="https://www.zhihu.com/equation?tex=\mathbf{w}_{k}(t)=\frac{1}{2}\left(\mathbf{w}^{(i)}(t)+\mathbf{w}^{(j)}(t)\right)
" alt="\mathbf{w}_{k}(t)=\frac{1}{2}\left(\mathbf{w}^{(i)}(t)+\mathbf{w}^{(j)}(t)\right)
" class="ee_img tr_noresize" eeimg="1">
之后方便演示采用的都是这种假设，假设客户 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 本地训练得到了模型 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}_k(t+E)" alt="\mathbf{w}_k(t+E)" class="ee_img tr_noresize" eeimg="1"> ，可以将其发送给相应的ES，对于重叠的 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 也不用多次通信，因为采用广播的形式。

之后每个ES对其覆盖区域 <img src="https://www.zhihu.com/equation?tex=C_i" alt="C_i" class="ee_img tr_noresize" eeimg="1"> 的客户端的模型进行更新聚合，每个ES对接收到的模型进行加权平均，其中权重取决于客户端k的位置(即客户端k是否位于重叠区域)。设 <img src="https://www.zhihu.com/equation?tex=S_{t+E}^{(i)}" alt="S_{t+E}^{(i)}" class="ee_img tr_noresize" eeimg="1"> 是将结果发送给ES <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的集合，ES对模型进行平均：

<img src="https://www.zhihu.com/equation?tex=\mathbf{w}^{(i)}(t+E)=\sum_{k \in S_{t+E}^{(i)}} \gamma_{k}^{(i)} \mathbf{w}_{k}(t+E)
" alt="\mathbf{w}^{(i)}(t+E)=\sum_{k \in S_{t+E}^{(i)}} \gamma_{k}^{(i)} \mathbf{w}_{k}(t+E)
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\gamma" alt="\gamma" class="ee_img tr_noresize" eeimg="1"> 就是权重参数，表示为：

<img src="https://www.zhihu.com/equation?tex=\gamma_{k}^{(i)}=\left\{\begin{array}{ll}
\alpha_{u} n_{k}, & k \in U_{i} \\
\alpha_{v} n_{k}, & k \in V_{i, j} \text { for any } j
\end{array} \quad \text { and } \quad \sum_{k \in S_{t+E}^{(i)}} \gamma_{k}^{(i)}=1\right.
" alt="\gamma_{k}^{(i)}=\left\{\begin{array}{ll}
\alpha_{u} n_{k}, & k \in U_{i} \\
\alpha_{v} n_{k}, & k \in V_{i, j} \text { for any } j
\end{array} \quad \text { and } \quad \sum_{k \in S_{t+E}^{(i)}} \gamma_{k}^{(i)}=1\right.
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\alpha_u" alt="\alpha_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 为权重参数，例如 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 更大则代表赋予重叠部分的节点更多的权重，若相等则针对该ES来说则和FedAvg一样。

当整体训练完毕后，再将整体的ES平均聚合：

<img src="https://www.zhihu.com/equation?tex=\mathbf{w}^{f}(T)=\frac{1}{L} \sum_{i=1}^{L} \mathbf{w}^{(i)}(T)
" alt="\mathbf{w}^{f}(T)=\frac{1}{L} \sum_{i=1}^{L} \mathbf{w}^{(i)}(T)
" class="ee_img tr_noresize" eeimg="1">
算法流程表示为：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220302165807560.png" alt="image-20220302165807560" style="zoom:50%;" />

如图所示，搭建一个FedMes系统，可以得到：
<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220302172536638.jpg" alt="image-20220302172536638" style="zoom:50%;" />

精简算法可以得到，客户端 <img src="https://www.zhihu.com/equation?tex=k\in C_i" alt="k\in C_i" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=t+1" alt="t+1" class="ee_img tr_noresize" eeimg="1"> 步的模型表示为：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220302193029532.png" alt="image-20220302193029532" style="zoom:50%;" />

对于多于2个的客户端，直接添加方程即可，由于客户端在任何位置都成立，所以理论上可以很好地拓展到移动的客户端。对于 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1"> 个覆盖量级的设备可以表示为：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220302194619873.png" alt="image-20220302194619873" style="zoom:50%;" />

- **延迟分析**

1、FedMes

假设模型的大小是 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 比特， <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 为cpu计算1比特需要的周期数， <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为cpu的每秒周期数（算力），设进行 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 个epoch计算需要计算 <img src="https://www.zhihu.com/equation?tex=d(E)" alt="d(E)" class="ee_img tr_noresize" eeimg="1"> 比特，那么每个客户端计算 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 个本地更新的时间 <img src="https://www.zhihu.com/equation?tex=t_{comp}" alt="t_{comp}" class="ee_img tr_noresize" eeimg="1"> 表示为：

<img src="https://www.zhihu.com/equation?tex=t_{comp}=\frac{c\times d(E)}{f}
" alt="t_{comp}=\frac{c\times d(E)}{f}
" class="ee_img tr_noresize" eeimg="1">
设 <img src="https://www.zhihu.com/equation?tex=B_u" alt="B_u" class="ee_img tr_noresize" eeimg="1"> 为上行带宽， <img src="https://www.zhihu.com/equation?tex=B_d" alt="B_d" class="ee_img tr_noresize" eeimg="1"> 为下行带宽，边缘服务器使用它向每个客户端发送聚合的全局模型。上行通道增益表示为 <img src="https://www.zhihu.com/equation?tex=h_u=g_u/d_{edge}^\eta" alt="h_u=g_u/d_{edge}^\eta" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=g_u" alt="g_u" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=\eta" alt="\eta" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=d_{edge}" alt="d_{edge}" class="ee_img tr_noresize" eeimg="1"> 表示快速衰落分量，路径损耗指数和ES和客户端之间的距离。同时下行信道增益表示为 <img src="https://www.zhihu.com/equation?tex=h_d=g_u/d_{edge}^\eta" alt="h_d=g_u/d_{edge}^\eta" class="ee_img tr_noresize" eeimg="1"> 。最后我们设定 <img src="https://www.zhihu.com/equation?tex=p_{client}" alt="p_{client}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=p_{edge}" alt="p_{edge}" class="ee_img tr_noresize" eeimg="1"> 为发射功率，所以客户端和边缘往返一次的时间 <img src="https://www.zhihu.com/equation?tex=t_{edge}" alt="t_{edge}" class="ee_img tr_noresize" eeimg="1"> 可以写为：

<img src="https://www.zhihu.com/equation?tex=t_{\text {edge }}=\frac{M}{B_{u} \log _{2}\left(1+\frac{h_{u} p_{\text {client }}}{B_{u} N_{0}}\right)}+\frac{M}{B_{d} \log _{2}\left(1+\frac{h_{d} p_{\text {edge }}}{B_{d} N_{0}}\right)}
" alt="t_{\text {edge }}=\frac{M}{B_{u} \log _{2}\left(1+\frac{h_{u} p_{\text {client }}}{B_{u} N_{0}}\right)}+\frac{M}{B_{d} \log _{2}\left(1+\frac{h_{d} p_{\text {edge }}}{B_{d} N_{0}}\right)}
" class="ee_img tr_noresize" eeimg="1">
左边是上行通信时间，可以看到是明显大于下行通信时间的。因为客户通常具有更小的发射功率。所以FedMes单轮的时间表示为：

<img src="https://www.zhihu.com/equation?tex=t_{FedMes}=t_{comp}+t_{edge}
" alt="t_{FedMes}=t_{comp}+t_{edge}
" class="ee_img tr_noresize" eeimg="1">
2、基于云的FL：

总的时间为：

<img src="https://www.zhihu.com/equation?tex=t_{CloudFL}=t_{comp}+t_{cloud}
" alt="t_{CloudFL}=t_{comp}+t_{cloud}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=t_{cloud}" alt="t_{cloud}" class="ee_img tr_noresize" eeimg="1"> 是客户端和云服务器之间的一次往返通信时间，云端和客户端的距离远大于边缘到客户端的距离，即 <img src="https://www.zhihu.com/equation?tex=d_{edge}<<d_{cloud}" alt="d_{edge}<<d_{cloud}" class="ee_img tr_noresize" eeimg="1"> 。所以FedMes时间更短。

3、基于层的FL

假设在 <img src="https://www.zhihu.com/equation?tex=T_{cloud}" alt="T_{cloud}" class="ee_img tr_noresize" eeimg="1"> 时间中进行了一次全局聚合，则一个全局时间可以表示为：

<img src="https://www.zhihu.com/equation?tex=t_{\text {Hier }}=\frac{1}{T_{\text {cloud }}}\left[\left(t_{\text {comp }}+t_{\text {edge }}\right)\left(T_{\text {cloud }}-1\right)+t_{\text {comp }}+t_{\text {cloud }}\right]
" alt="t_{\text {Hier }}=\frac{1}{T_{\text {cloud }}}\left[\left(t_{\text {comp }}+t_{\text {edge }}\right)\left(T_{\text {cloud }}-1\right)+t_{\text {comp }}+t_{\text {cloud }}\right]
" class="ee_img tr_noresize" eeimg="1">
可以将 <img src="https://www.zhihu.com/equation?tex=t_{FedMes}" alt="t_{FedMes}" class="ee_img tr_noresize" eeimg="1"> 改写为 <img src="https://www.zhihu.com/equation?tex=t_{\text {Hier }}=\frac{1}{T_{\text {cloud }}}\left[\left(t_{\text {comp }}+t_{\text {edge }}\right)\left(T_{\text {cloud }}-1\right)+t_{\text {comp }}+t_{\text {edge }}\right]" alt="t_{\text {Hier }}=\frac{1}{T_{\text {cloud }}}\left[\left(t_{\text {comp }}+t_{\text {edge }}\right)\left(T_{\text {cloud }}-1\right)+t_{\text {comp }}+t_{\text {edge }}\right]" class="ee_img tr_noresize" eeimg="1"> ，所以其时间是远远小于分层FL的。

综上，分散的FL的延迟其实取决于网络拓扑，即**客户端之间怎么连接**。

## Theoretical Result

假设1：对于所有的 <img src="https://www.zhihu.com/equation?tex=k\in\{1,2,...,K\}" alt="k\in\{1,2,...,K\}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=F_k(\mathbf{w})" alt="F_k(\mathbf{w})" class="ee_img tr_noresize" eeimg="1"> 对于 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}" alt="\mathbf{w}" class="ee_img tr_noresize" eeimg="1"> 来说是 <img src="https://www.zhihu.com/equation?tex=\beta" alt="\beta" class="ee_img tr_noresize" eeimg="1"> 平滑的。对于任意的 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}" alt="\mathbf{w}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}^ \prime" alt="\mathbf{w}^ \prime" class="ee_img tr_noresize" eeimg="1"> 来说，有 <img src="https://www.zhihu.com/equation?tex=F_{k}(\mathbf{w}) \leq F_{k}\left(\mathbf{w}^{\prime}\right)+\left(\mathbf{w}-\mathbf{w}^{\prime}\right) \nabla F_{k}\left(\mathbf{w}^{\prime}\right)+\frac{\beta}{2}\left\|\mathbf{w}-\mathbf{w}^{\prime}\right\|^{2}" alt="F_{k}(\mathbf{w}) \leq F_{k}\left(\mathbf{w}^{\prime}\right)+\left(\mathbf{w}-\mathbf{w}^{\prime}\right) \nabla F_{k}\left(\mathbf{w}^{\prime}\right)+\frac{\beta}{2}\left\|\mathbf{w}-\mathbf{w}^{\prime}\right\|^{2}" class="ee_img tr_noresize" eeimg="1"> 

假设2：对于所有的 <img src="https://www.zhihu.com/equation?tex=k\in\{1,2,...,K\}" alt="k\in\{1,2,...,K\}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=F_k(\mathbf{w})" alt="F_k(\mathbf{w})" class="ee_img tr_noresize" eeimg="1"> 是 <img src="https://www.zhihu.com/equation?tex=\mu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 强凸的。对于任意的 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}" alt="\mathbf{w}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}^ \prime" alt="\mathbf{w}^ \prime" class="ee_img tr_noresize" eeimg="1"> 来说， <img src="https://www.zhihu.com/equation?tex=F_{k}(\mathbf{w}) \geq F_{k}\left(\mathbf{w}^{\prime}\right)+\left(\mathbf{w}-\mathbf{w}^{\prime}\right) \nabla F_{k}\left(\mathbf{w}^{\prime}\right)+\frac{\mu}{2}\left\|\mathbf{w}-\mathbf{w}^{\prime}\right\|^{2}" alt="F_{k}(\mathbf{w}) \geq F_{k}\left(\mathbf{w}^{\prime}\right)+\left(\mathbf{w}-\mathbf{w}^{\prime}\right) \nabla F_{k}\left(\mathbf{w}^{\prime}\right)+\frac{\mu}{2}\left\|\mathbf{w}-\mathbf{w}^{\prime}\right\|^{2}" class="ee_img tr_noresize" eeimg="1"> 。

假设3：每个设备中的随机梯度是无偏的，其方差是有界的，也就是 <img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\xi}[\nabla F_{k}(\mathbf{w}, \xi)]=\nabla F_k(\mathbf{w})" alt="\mathbb{E}_{\xi}[\nabla F_{k}(\mathbf{w}, \xi)]=\nabla F_k(\mathbf{w})" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\xi}\left[\left\|\nabla F_{k}(\mathbf{w}, \xi)-\nabla F_{k}(\mathbf{w})\right\|\right] \leq \sigma_{k}^{2}" alt="\mathbb{E}_{\xi}\left[\left\|\nabla F_{k}(\mathbf{w}, \xi)-\nabla F_{k}(\mathbf{w})\right\|\right] \leq \sigma_{k}^{2}" class="ee_img tr_noresize" eeimg="1"> 

假设4：随机梯度的期望平方范数是一致有界的， <img src="https://www.zhihu.com/equation?tex=\mathbb{E}_{\xi}\left[\left\|\nabla F_{k}(\mathbf{w}, \xi)\right\|^2\right] \leq G^{2}" alt="\mathbb{E}_{\xi}\left[\left\|\nabla F_{k}(\mathbf{w}, \xi)\right\|^2\right] \leq G^{2}" class="ee_img tr_noresize" eeimg="1"> 。

为了便于分析，我们假设系统中的所有客户都参与了每一轮全局聚合：

<img src="https://www.zhihu.com/equation?tex=\mathbf{v}_{k}(t+1)=\mathbf{w}_{k}(t)-\eta_{t} \nabla F_{k}\left(\mathbf{w}_{k}(t), \xi_{k}(t)\right)
" alt="\mathbf{v}_{k}(t+1)=\mathbf{w}_{k}(t)-\eta_{t} \nabla F_{k}\left(\mathbf{w}_{k}(t), \xi_{k}(t)\right)
" class="ee_img tr_noresize" eeimg="1">
那么上一节的11式可以写为：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304085123840.png" alt="image-20220304085123840" style="zoom:50%;" />

设 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}^*" alt="\mathbf{w}^*" class="ee_img tr_noresize" eeimg="1"> 是最优解，我们得到：

<img src="https://www.zhihu.com/equation?tex=F\left(\mathbf{w}^{f}(T)\right)-F\left(\mathbf{w}^{*}\right) \leq\left(\mathbf{w}^{f}(T)-\mathbf{w}^{*}\right)^{T} \nabla F\left(\mathbf{w}^{*}\right)+\frac{\beta}{2}\left\|\mathbf{w}^{f}(T)-\mathbf{w}^{*}\right\|^{2}
" alt="F\left(\mathbf{w}^{f}(T)\right)-F\left(\mathbf{w}^{*}\right) \leq\left(\mathbf{w}^{f}(T)-\mathbf{w}^{*}\right)^{T} \nabla F\left(\mathbf{w}^{*}\right)+\frac{\beta}{2}\left\|\mathbf{w}^{f}(T)-\mathbf{w}^{*}\right\|^{2}
" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=\mathbf{w}^f(T)" alt="\mathbf{w}^f(T)" class="ee_img tr_noresize" eeimg="1"> 是边缘端一起聚合得到的最终模型。假设 <img src="https://www.zhihu.com/equation?tex=\alpha_u=2\alpha_v" alt="\alpha_u=2\alpha_v" class="ee_img tr_noresize" eeimg="1"> 并且所有客户端都有相同数量的数据样本。然后通过两边取期望：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304091943215.png" alt="image-20220304091943215" style="zoom:50%;" />

其中 <img src="https://www.zhihu.com/equation?tex=\overline{\mathbf{v}}(t)=\frac{1}{K} \sum_{k=1}^{K} \mathbf{v}_{k}(t)" alt="\overline{\mathbf{v}}(t)=\frac{1}{K} \sum_{k=1}^{K} \mathbf{v}_{k}(t)" class="ee_img tr_noresize" eeimg="1"> ，前期文献证明：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304092421443.png" alt="image-20220304092421443" style="zoom:50%;" />

我们给出了FedMes的收敛界限：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304094505231.png" alt="image-20220304094505231" style="zoom:50%;" />

可以看到收敛界限包含两部分，第1部分和传统FL相似，随着时间步长 <img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 的增加而逐渐趋于0，第2项是每个客户的模型与时间步T−1的平均模型之间的差距。在传统的基于云的FL方案中，所有客户端的模型在每一轮**全局回合都是同步**的，第二项(22)也随着T的增加而趋于零。

更具体地说，当所有客户端的模型收敛于同一模型时，这一项趋于零。这里给出了更细致的界限：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304101616435.png" alt="image-20220304101616435" style="zoom:50%;" />

左边的一项随着 <img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 增大逐渐趋于0，右边的可以看做误差项。由于 <img src="https://www.zhihu.com/equation?tex=K=(u+v)L" alt="K=(u+v)L" class="ee_img tr_noresize" eeimg="1"> ，所以 <img src="https://www.zhihu.com/equation?tex=u=KL-v" alt="u=KL-v" class="ee_img tr_noresize" eeimg="1"> 说明是 <img src="https://www.zhihu.com/equation?tex=v" alt="v" class="ee_img tr_noresize" eeimg="1"> 的一个递减函数，则重叠越多，误差越小。

## Experimental Result

数据集：MNIST、Cifar

模型：CNN、VGG-11

对比模型：传统HFL，本地训练 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1"> 轮次，在边缘端聚合 <img src="https://www.zhihu.com/equation?tex=T_{cloud}" alt="T_{cloud}" class="ee_img tr_noresize" eeimg="1"> 次，然后在云端聚合一次。 <img src="https://www.zhihu.com/equation?tex=T_{cloud}=1" alt="T_{cloud}=1" class="ee_img tr_noresize" eeimg="1"> 代表传统的FedAvg模型， <img src="https://www.zhihu.com/equation?tex=T_{cloud}=\infty" alt="T_{cloud}=\infty" class="ee_img tr_noresize" eeimg="1"> ，代表没有重复客户端，最后再进行边缘到云的聚合。

实验装置：

 <img src="https://www.zhihu.com/equation?tex=L=3" alt="L=3" class="ee_img tr_noresize" eeimg="1"> 个边缘并且 <img src="https://www.zhihu.com/equation?tex=K=90" alt="K=90" class="ee_img tr_noresize" eeimg="1"> 客户端，我们考虑几何结构为 <img src="https://www.zhihu.com/equation?tex=|U_1|=|U_2|=|U_3|=u" alt="|U_1|=|U_2|=|U_3|=u" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=|V_{1,2}|=|V_{2,3}|=|V_{1,3}|=v" alt="|V_{1,2}|=|V_{2,3}|=|V_{1,3}|=v" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=u+v=K/3=30" alt="u+v=K/3=30" class="ee_img tr_noresize" eeimg="1"> ，每次边缘聚合器的聚合选择其覆盖中的 <img src="https://www.zhihu.com/equation?tex=m=20" alt="m=20" class="ee_img tr_noresize" eeimg="1"> 个设备，如果没有重合的话一共60个设备。当 <img src="https://www.zhihu.com/equation?tex=v>0" alt="v>0" class="ee_img tr_noresize" eeimg="1"> 时，每个客户端都可以访问 <img src="https://www.zhihu.com/equation?tex=u+2v" alt="u+2v" class="ee_img tr_noresize" eeimg="1"> 个客户端，所以从非重叠端 <img src="https://www.zhihu.com/equation?tex=U_i" alt="U_i" class="ee_img tr_noresize" eeimg="1"> 中选择 <img src="https://www.zhihu.com/equation?tex=\frac{um}{u+2v}" alt="\frac{um}{u+2v}" class="ee_img tr_noresize" eeimg="1"> 个客户端，在重叠部分选择 <img src="https://www.zhihu.com/equation?tex=\frac{2vm}{u+2v}" alt="\frac{2vm}{u+2v}" class="ee_img tr_noresize" eeimg="1"> 客户端。假设选的很平均即每个重合端选择 <img src="https://www.zhihu.com/equation?tex=\frac{vm}{u+2v}" alt="\frac{vm}{u+2v}" class="ee_img tr_noresize" eeimg="1"> ，所以该方案选择的客户端数量小于无重叠方案。

我们设置 <img src="https://www.zhihu.com/equation?tex=t_{edge}/t_{comp}=10" alt="t_{edge}/t_{comp}=10" class="ee_img tr_noresize" eeimg="1"> ，并将 <img src="https://www.zhihu.com/equation?tex=t_{edge}" alt="t_{edge}" class="ee_img tr_noresize" eeimg="1"> 归一为1，定义 <img src="https://www.zhihu.com/equation?tex=t_c=t_{cloud}/t_{edge}>>1" alt="t_c=t_{cloud}/t_{edge}>>1" class="ee_img tr_noresize" eeimg="1"> ，设置学习率为0.1，动量项为（历史积累）0.9。

对于noniid设置了3种情况：

1、客户端iid，边缘iid

2、客户端noniid，边缘iid：先按iid分三份，然后每个客户端noniid分配

3、全都noniid

对于前两种情况，我们只考虑 <img src="https://www.zhihu.com/equation?tex=\alpha_u=\alpha_v" alt="\alpha_u=\alpha_v" class="ee_img tr_noresize" eeimg="1"> 的情况，因为 <img src="https://www.zhihu.com/equation?tex=\alpha_u=\alpha_v" alt="\alpha_u=\alpha_v" class="ee_img tr_noresize" eeimg="1"> 的方案在这些情况下已经显示出最优的性能（在FedAvg中应用便是这种）。最后一种情况考虑多种 <img src="https://www.zhihu.com/equation?tex=\alpha_u" alt="\alpha_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 的情况。

神经网络的结果：

1、客户端iid，边缘iid

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304152727800.png" alt="image-20220304152727800" style="zoom:50%;" />

前2个数据集FedMes其参数不同时和无重叠的情况相差不大，但在Cifar中FedMes明显要好一些（准确率更高）。对于 <img src="https://www.zhihu.com/equation?tex=T_{cloud}=1" alt="T_{cloud}=1" class="ee_img tr_noresize" eeimg="1"> 基于云的情况，最终也可以达到理想的情况但是消耗更高。 <img src="https://www.zhihu.com/equation?tex=T_{cloud}" alt="T_{cloud}" class="ee_img tr_noresize" eeimg="1"> 变大时效率会提升，但当趋于 <img src="https://www.zhihu.com/equation?tex=\infty" alt="\infty" class="ee_img tr_noresize" eeimg="1"> 时其和无重叠的算法一（黄色线）情况一样，所以精度必会下降。

2、客户端noniid，边缘iid

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304155528169.png" alt="image-20220304155528169" style="zoom:50%;" />

每种方案的准确率都会下降，没有重叠的情况下提出的算法1与其他FedMes有很大差距。

3、均noniid的情况

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304160732689.png" alt="image-20220304160732689" style="zoom:50%;" />

考虑了 <img src="https://www.zhihu.com/equation?tex=\alpha_u=\alpha_v" alt="\alpha_u=\alpha_v" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\alpha_u<\alpha_v" alt="\alpha_u<\alpha_v" class="ee_img tr_noresize" eeimg="1"> 2种情况。比较有趣的是HFL在每一次云聚合之前都会降低，这是因为每一个边缘部分的数据集都是不均衡的所以可能在同步之前降低性能。没有重叠的算法准确率最低。对重叠部分关注度越高（ <img src="https://www.zhihu.com/equation?tex=\alpha_u<\alpha_v" alt="\alpha_u<\alpha_v" class="ee_img tr_noresize" eeimg="1"> ）准确率越高。

4、考虑变化的 <img src="https://www.zhihu.com/equation?tex=t_c=t_{cloud}/t_{edge}" alt="t_c=t_{cloud}/t_{edge}" class="ee_img tr_noresize" eeimg="1"> 

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304164740407.png" alt="image-20220304164740407" style="zoom:50%;" />

当 <img src="https://www.zhihu.com/equation?tex=t_c" alt="t_c" class="ee_img tr_noresize" eeimg="1"> 变大时，FedMes表现得非常好。

5、测试精度与全局轮次

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304165958131.png" alt="image-20220304165958131" style="zoom:50%;" />

先观察误差项 <img src="https://www.zhihu.com/equation?tex=Q(T)" alt="Q(T)" class="ee_img tr_noresize" eeimg="1"> ，可以看到最终减小到一个常数。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304171901463.jpg" alt="image-20220304171901463" style="zoom:50%;" />

可以看到大部分情况(u = 0, v = 30)和(u = 20, v = 10)的FedMes最终精度与基于云的系统(紫色线)相同。在图（i）中没有，误差即为理论所计算的。前提是数据分布noniid且相对复杂时。

6、 <img src="https://www.zhihu.com/equation?tex=\alpha_u" alt="\alpha_u" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 的影响

图5分析了 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 大一些有利于FedMes的速度。我们在此多做了一些实验：如果 <img src="https://www.zhihu.com/equation?tex=\alpha_v" alt="\alpha_v" class="ee_img tr_noresize" eeimg="1"> 过小则不利于学习效率，如果过大则可能导致整体准确率下降。测验展示，对于MNIST来说 <img src="https://www.zhihu.com/equation?tex=\alpha_v=1.3\alpha_u" alt="\alpha_v=1.3\alpha_u" class="ee_img tr_noresize" eeimg="1"> 达到最好的效果。对于FMNIST和Cifar来说 <img src="https://www.zhihu.com/equation?tex=\alpha_v=1.5\alpha_u" alt="\alpha_v=1.5\alpha_u" class="ee_img tr_noresize" eeimg="1"> 达到最好。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304174019802.jpg" alt="image-20220304174019802" style="zoom:50%;" />

7、多重重叠的实验

考虑不止有设备存在于2个单元下的情况，引入了量 <img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1"> 如图：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304174540488.png" alt="image-20220304174540488" style="zoom:50%;" />

客户端的数量写为： <img src="https://www.zhihu.com/equation?tex=K=3(u+v)+w" alt="K=3(u+v)+w" class="ee_img tr_noresize" eeimg="1"> ，但实际上只要重叠部分有更多的客户端便可以加快训练速度。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304174900746.png" alt="image-20220304174900746" style="zoom:50%;" />

8、非对称单元的实验

将每个单元内的客户端数量设置为：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304175037273.png" alt="image-20220304175037273" style="zoom:50%;" />

得到的结果：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304175144084.png" alt="image-20220304175144084" style="zoom:50%;" />

结果和对称情况相同。

9、多个单元的拓展情况

假设有四个单元，设 <img src="https://www.zhihu.com/equation?tex=|U_i|=u" alt="|U_i|=u" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=|V_{1,2}|=|V_{2,3}|=|V_{3,4}|=|V_{4,1}|=v" alt="|V_{1,2}|=|V_{2,3}|=|V_{3,4}|=|V_{4,1}|=v" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=|V_{1,3}|=|V_{2,4}|=0" alt="|V_{1,3}|=|V_{2,4}|=0" class="ee_img tr_noresize" eeimg="1"> 。结果：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304175900463.jpg" alt="image-20220304175900463" style="zoom:50%;" />

测试精度与时间的关系：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMes/image-20220304180043031.png" alt="image-20220304180043031" style="zoom:50%;" />

因为云的大开销，所以紫线最低，其他方案都大致相同，但FedMes在双noniid的情况下优势明显。