# FedMD: Heterogenous Federated Learning via Model Distillation学习笔记

[toc]

## Abstract

普通的联邦学习可以在不侵犯参与者隐私的情况下，创建强大的集中模型，但它没有考虑每个参与者独立设计自己模型的情况。

本篇采用了迁移学习和知识蒸馏来开发一个通用的框架，当每个agent不仅拥有自己的私有数据，而且拥有唯一设计的模型时，可以进行联邦学习。

---

## 1 Introduction

诞生原因：解决用户的隐私问题，但学习过程中各个方面都存在着异质性，当每个参与者的**带宽和计算能力**不同时，系统存在异质性，还存在客户端本身**数据异质性**。

本文，我们关注一种不同类型的异质性：**<u>*局部模型的差异*</u>**。在最初的联合框架中，所有用户都必须就集中式模型的特定体系结构达成一致。当参与者是数以百万计的低容量设备(如手机)时，这是一个合理的假设。本文中，我们转而探索**联邦框架的扩展**，该扩展在面向业务的环境中是现实的，在这种环境中，每个参与者都有能力并希望设计自己独特的模型。

例如，当几个医疗机构协作而不共享私人数据时，它们可能需要构建自己的模型来满足不同的规范。出于隐私和知识产权的考虑，他们可能不愿意分享他们的模型的细节。

标准做法是客户只用自己的数据训练自己的模型，但如果可以在不侵犯他人隐私的情况下，利用其他客户的数据是很好的，但每个参与者的模型都算一个黑盒。

- 全模型异构的关键是通信。特别是，必须有一种**转换协议**，使深度网络能够在不共享数据或模型体系结构的情况下理解他人的知识。原则上，机器应该能够学习适应任何特定用例的最佳通信协议。作为这个方向的第一步，我们采用了一个**基于知识蒸馏**的更透明的框架来解决这个问题。

- 由于私有数据集量过小，所以从大型公共数据集进行迁移学习是必要的。主要通过2种方式借助迁移学习的力量，（1）在进入协作之前，每个模型首先在公共数据上，然后在它自己的私有数据上接受完全训练。（2）黑箱模型基于它们在公共数据集样本上的输出类得分进行通信。该步骤通过知识蒸馏实现。

***贡献***：实现了FedMD，可以让参与者独立设计他们的模型。我们的集中式服务器不控制这些模型的体系结构，只需要有限的黑盒访问。通讯协议通过迁移学习和知识蒸馏实现。最终测试结果表明，与没有协作相比，该模型在局部模型性能方面显著提高。

---

## 2 Methods

### 2.1 Problem definition

设有 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> 个参与者，每个参与者都有非常小的数据集 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{k}:=\left\{\left(x_{i}^{k}, y_{i}\right)\right\}_{i=1}^{N_{k}}" alt="\mathcal{D}_{k}:=\left\{\left(x_{i}^{k}, y_{i}\right)\right\}_{i=1}^{N_{k}}" class="ee_img tr_noresize" eeimg="1"> ，有可能是IID或Non-IID，还有一个大型的共用数据集 <img src="https://www.zhihu.com/equation?tex=\mathcal{D}_{0}:=\left\{\left(x_{i}^{0}, y_{i}\right)\right\}_{i=1}^{N_{0}}" alt="\mathcal{D}_{0}:=\left\{\left(x_{i}^{0}, y_{i}\right)\right\}_{i=1}^{N_{0}}" class="ee_img tr_noresize" eeimg="1"> ，这个数据集每个参与者都可以访问，每个参与者独立设计自己的模型 <img src="https://www.zhihu.com/equation?tex=f_k" alt="f_k" class="ee_img tr_noresize" eeimg="1"> 实现一个分类任务，**参与者不需要共享超参数**，但可以用一个协作框架，利用本地可访问的 <img src="https://www.zhihu.com/equation?tex=D_0" alt="D_0" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=D_k" alt="D_k" class="ee_img tr_noresize" eeimg="1"> 来改进 <img src="https://www.zhihu.com/equation?tex=f_k" alt="f_k" class="ee_img tr_noresize" eeimg="1"> 的效果。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMD：基于模型蒸馏的异构联合学习/image-20211225110215388.png" alt="image-20211225110215388" style="zoom:50%;" />

> 每个代理都拥有一个私有数据集和一个独特设计的模型。为了在不泄露数据的情况下进行通信和协作，代理需要将他们学到的知识转换为标准格式。中央服务器收集这些知识，计算出分布在网络上的共识。在这项工作中，翻译器是使用知识蒸馏来实现的。

### 2.2 The framework for heterogeneous federated learning

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMD：基于模型蒸馏的异构联合学习/image-20211225110520971.png" alt="image-20211225110520971" style="zoom:50%;" />

**<u>*迁移学习*</u>**：在参与者开始协作阶段之前，其模型必须首先经历整个迁移学习过程。**它将在公共数据集上完全训练，然后在它自己的私有数据上训练。**因此，未来的任何改进都将与此基准进行比较。

***<u>交流过程</u>***：

1. 迁移学习结束后，每个设备都具备自己的 <img src="https://www.zhihu.com/equation?tex=f_k" alt="f_k" class="ee_img tr_noresize" eeimg="1"> ，利用公共数据计算 <img src="https://www.zhihu.com/equation?tex=f_k(x_i^0)" alt="f_k(x_i^0)" class="ee_img tr_noresize" eeimg="1"> 预测结果分数，并将其推送给服务器。
2. 中心服务器计算平均分数 <img src="https://www.zhihu.com/equation?tex=\tilde{f}\left(x_{i}^{0}\right)=\frac{1}{m} \sum_{k} f_{k}\left(x_{i}^{0}\right)" alt="\tilde{f}\left(x_{i}^{0}\right)=\frac{1}{m} \sum_{k} f_{k}\left(x_{i}^{0}\right)" class="ee_img tr_noresize" eeimg="1"> 。
3. 客户端下载平均分数 <img src="https://www.zhihu.com/equation?tex=\tilde{f}\left(x_{i}^{0}\right)" alt="\tilde{f}\left(x_{i}^{0}\right)" class="ee_img tr_noresize" eeimg="1"> 。
4. 知识蒸馏，每个客户端用模型蒸馏在共享数据集去拟合这个平均分数，即各个模型去学习全局共识。即每个模型用 <img src="https://www.zhihu.com/equation?tex=f_k" alt="f_k" class="ee_img tr_noresize" eeimg="1"> 在数据 <img src="https://www.zhihu.com/equation?tex=D_0" alt="D_0" class="ee_img tr_noresize" eeimg="1"> 上去逼近 <img src="https://www.zhihu.com/equation?tex=\tilde{f}" alt="\tilde{f}" class="ee_img tr_noresize" eeimg="1"> 。
5. 每个客户端设备再利用自己的私有数据训练 <img src="https://www.zhihu.com/equation?tex=f_k" alt="f_k" class="ee_img tr_noresize" eeimg="1"> 几个epochs。

一个参与者的知识可以被其他参与者理解，而无需共享其私有数据或模型体系结构。在第1步时，可以选择较少的部分数据集 <img src="https://www.zhihu.com/equation?tex=d_{j} \subset \mathcal{D}_{0}" alt="d_{j} \subset \mathcal{D}_{0}" class="ee_img tr_noresize" eeimg="1"> 进行训练。

> 1、第1步里的预测结果分数是指不经过softmax的结果。
>
> 2、通信阶段选择的子数据集大小大约为5000。
>
> 3、第2步中的值理论上也可以用加权平均值进行计算： <img src="https://www.zhihu.com/equation?tex=\tilde{f}\left(x_{i}^{0}\right)=\sum_{k} c_{k} f_{k}\left(x_{i}^{0}\right)" alt="\tilde{f}\left(x_{i}^{0}\right)=\sum_{k} c_{k} f_{k}\left(x_{i}^{0}\right)" class="ee_img tr_noresize" eeimg="1"> ，一个例子是在CIFAR的案例中，我们略微抑制了两个较弱的模型(0和9)的贡献。

---

## 3 Result

设置了2种环境：

1. 公共数据是MNIST，私有数据是FEMNIST的子集，考虑了IID和Non-IID两种情况。
2. 共有数据是Cifar10，私有数据是Cifar100的子集。Cifar100在20个大类下分了100个小类。对于Non-IID情况，每个设备有一个来自大类的小类，但最终测试需要将任意其他小类分到正确的大类中，例：一名在训练期间只见过狼的参与者被期望正确地将狮子归类为大型肉食动物。

在每个环境中，10名参与者设计独一无二的卷积网络，其通道数和层数都不大相同，如下表：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMD：基于模型蒸馏的异构联合学习/image-20211225165050199.png" alt="image-20211225165050199" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMD：基于模型蒸馏的异构联合学习/image-20211225165104783.png" alt="image-20211225165104783" style="zoom:50%;" />

首先在公共数据集上训练，MNIST准确率在99%左右，Cifar10在76%左右。然后在自己的私有数据集上训练，至此实现迁移学习。

通信过程中，选取Adam学习器，学习率0.001，每一次通信过程中，选取 <img src="https://www.zhihu.com/equation?tex=d_{j} \subset \mathcal{D}_{0}" alt="d_{j} \subset \mathcal{D}_{0}" class="ee_img tr_noresize" eeimg="1"> 大小为5000。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/FedMD：基于模型蒸馏的异构联合学习/image-20211225173208330.png" alt="image-20211225173208330" style="zoom:50%;" />

**左边的虚线**表示一个模型在使用公共数据集和它自己的小型私有数据集进行完全迁移学习后的测试精度。**右边的虚线**表示如果私有数据都被破解并开放，测试出来的结果。所体现的线就代表，经过知识蒸馏之后的训练结果。

---

## 4 Discussion and conclusion

我们提出了FedMD，一个能够为独立设计的模型实现联合学习的框架。我们的框架是基于知识蒸馏和测试工作的各种任务和数据集。

很多的联邦学习框架中，客户端都是发送模型参数给服务端从而在服务端聚合成一个全局的模型。在这篇论文却提出了发送模型在**公共数据集上预测分数**，在服务端上集成这些分数得到一个**全局共识**，客户端模型再去学习这些共识。

通过此方式的好处有：

1. 每个客户端可以根据自身条件训练出适合自己的模型，而不必全部客户端的模型都一样。
2. 有文章指出可以偷取模型更新梯度的情况下可以还原出训练数据。所以如果发送模型参数显然会增加数据隐私泄露的风险，而发送预测分数则不会出现这样的风险。
3. 减少传输的数据量。

不足：

每个用户都要牺牲自己的一部分隐私数据。