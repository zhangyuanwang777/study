# DPO阅读笔记

[DPO算法实现](https://github.com/eric-mitchell/direct-preference-optimization)，[使用 DPO 微调 Llama 2](https://huggingface.co/blog/zh/dpo-trl)，[使用 DPO 微调 Llama 2 的完整代码](https://github.com/lvwerra/trl/tree/main/examples/research_projects/stack_llama_2) 

## 什么是RLHF

人类反馈的强化学习 RLHF（Reinforcement Learning from Human Feedback）是一种微调范式，可以使用预训练模型作为 RLHF 的基础模型，微调之后可以使模型的输出更加符合人类的需求。

+ 为什么需要人类反馈 HF ？

  如果仅仅使用普通的文本评价指标（如 BLEU 分数、 ROUGE 分数）来衡量模型的优劣，最终生成的内容很难满足人类的需求，而 RLHF 可以使得在一般文本数据语料库上训练的语言模型能和复杂的人类价值观对齐。

+ 为什么要使用强化学习 RL ？

  强化学习可以通过奖励函数来调节自身的行为，奖励函数的设置有两种情况，可以每进行一个步骤就进行奖励（Reward），也可以在所有步骤进行完之后再进行奖励。第一种情况类似监督学习，而第二种情况强化学习更加擅长。

  <img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115092311564.png" alt="image-20240115092311564" style="zoom:80%;" />

  <img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115092253407.png" alt="image-20240115092253407" style="zoom:80%;" />

  对于问答模型而言，输入和输出都是一个句子，每个句子都是由单个词组成，我们很难为每个词都设置相应的标准答案，而只能针对一个句子来设置。因此该过程就类似与强化学习中多个动作步骤的优化学习

RLHF 运作过程主要包含以下几个方面：

### 数据集

问答数据集

### 监督式微调 Supervised Fine-Tuning（SFT）

SFT的主要目的是使pre-train之后的模型学会以对话的形式返回结果，而不是去做单纯的文本生成，一般来说使用收集的高质量Instruction数据进行训练，是RLHF流程的起始步骤。

### 训练奖励模型 Training a Reward Modedl（RM）

基于人类的偏好和标注，来训练一个能模拟人偏好的打分模型

### 强化学习 Reinforcement Learning（RL）

在前面的 SFT 模型的基础上，借助 RM 提供反馈，来不断通过 PPO 的强化学习框架来调整模型的行为

------



## 一种深度强化学习 PPO 

[李宏毅强化学习-YouTube]:https://www.youtube.com/watch?v=z95ZYgPgXOY
[李宏毅强化学习-BIibili]:https://www.bilibili.com/video/BV1MW411w79n/
[网友的笔记-1]:https://jianshu.com/p/9f113adc0c50
[网友的笔记-2]:https://zhuanlan.zhihu.com/p/468828804

Policy 是一个神经网络，用 $\pi$ 来表示，其中包含参数 $\theta$ 。

- 输入：当前所处的化境或状态（可以是向量或矩阵）
- 输出：采取不同行为的概率（具体采取哪个行为是随机的）

我们所要训练的就是参数 $\theta$ 。下面以游戏为例子讲解 PPO 。

### Actor-Environment-Reward

每一个所处的环境用 $s_n$ 来表示，每采取的一个动作用 $a_n$ 来表示，每一个动作之后会得到一个奖励，将奖励加总，得到最终的奖励 R ，R 就表示这一套策略所得到的奖励。这一套策略也就是所有环境和动作的集合，用 Trajectory 来表示，每一个 Trajectory 都可以计算其发生的概率 $p_{\theta}(\tau)$ 。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115162728544.png" alt="image-20240115162728544" style="zoom:80%;" />

对于 Reward ，我们需要做的就是调整 Actor 里面的参数，使得 Reward 达到最大，由于每一个 Trajectory 都是以一定的几率发生，因此将每个 Trajectory 得到的 Reward 加权即可得到最终的Expected Reward ，用 $\bar{R}_{\theta}$（或者$E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]$）来表示。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115163412562.png" alt="image-20240115163412562" style="zoom:80%;" />

### 最大化 Expected Reward

使用**梯度下降**更新参数，以最大化 Expected Reward。通过对 Expected Reward 求导，经过变形可以得到对数形式的结果。该结果可以看作一个新的期望形式，再将该期望写成累积求和的形式。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115164534985.png" alt="image-20240115164534985" style="zoom:80%;" />

其中：
$$
\nabla logp_{\theta}(\tau ^n)=\nabla \{logp(s_1)+\sum_{t=1}^{T^n}logp_{\theta}(a_t^n|s_t^n)+\sum_{t=1}^{T^n}logp(s_{t+1}^n|s_t^n,a_t^n)\}
\\=\sum_{t=1}^{T^n} \nabla logp_{\theta}(a_t^n|s_t^n)
$$
观察最终的结果，可以发现这就是一种加权的交叉熵损失，权重就是 $R(\tau ^n)$ 。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115174256835.png" alt="image-20240115174256835" style="zoom:80%;" />

### 实际当中的参数更新

在实际过程当中，随机设置模型的参数，再使用该模型与实际环境交互，得到多组 $\tau$ ，由于每次采取的决策是随机的，因此每个 $\tau$ 也都是不同的。

这样就可以计算得到 $\bar{R}_{\theta}$ ，使用梯度下降更新参数 $\theta$ 之后，再将模型与环境交互，循环上述步骤直至梯度足够小。

需要注意的是，每一轮参数更新过程中，所使用的 $\tau$ 并不共享，即该轮的 $\tau$ 只是用一次就丢弃了。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115171713986.png" alt="image-20240115171713986" style="zoom:80%;" />

实际当中根据不同的任务，可能需要进行不同的调整，下面是一些调整的小技巧：

#### 增加一个基线

如果所有的 Reward 都是正数，这样可能会由于采样的不完全，导致结果与预期相反。下图的例子指出，如果a没有被采样到，那么a最终被采样的概率就会下降（尽管其 Reward 很大）：

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115174927703.png" alt="image-20240115174927703" style="zoom:80%;" />

因此将梯度更新公式修改为：

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115175241855.png" alt="image-20240115175241855" style="zoom:80%;" />

#### 计算每个 Actor 的 Reward

每一个轮次（episode）当中，不同的 Actor 有好有坏，实际当中不应该只看最终的平均奖励。因此我们将当前 Actor 以及之后的 Reward 加总起来，作为当前 Actor 的 Reward（这里就认为当前的 Actor 会对将来产生影响，而不会对过去产生影响）。对梯度更新公式做如下修改：

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115182646880.png" alt="image-20240115182646880" style="zoom:80%;" />

修改之后的公式，对不同时间步的交叉熵（对数概率）使用不同的权重加权。

更进一步，考虑到不同时刻的动作对于后续状态的影响力会逐渐减小，将 Reward r 逐步乘以一个小于1的权重 $\gamma$ ，修改后的公式如下：

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115183129106.png" alt="image-20240115183129106" style="zoom:80%;" />

我们将 $\sum_{t^{'}=t}^{T_n} \gamma^{t^{'}-t} r_{t^{'}}^n - b$ 命为 $A^{\theta}(s_t,a_t)$ ，这表示每一个状态 $s_t$ 下所执行的动作 $a_t$ 的真实的 Reward。

### On-policy 与 Off-policy

- 与环境交互、收集数据，更新参数，这些步骤所使用的 Policy 是同一个，那么就称为 **On-policy**。
- 与环境交互、收集数据的 Policy ，更新参数的 Policy 不是同一个，那么称为 **Off-policy**。

上文提到的 Policy 就是 On-policy 。但是这种方式需要重复收集数据，耗时很长，下面我们将其修改为 Off-policy 。由于Off-policy 中 Policy 不同，导致数据所服从的分布与我们所要更新参数的分布是不同的，这样就引出了 Importance Sampling 的概念。

#### Importance Sampling

我们想要计算 $E_{x \sim p}[f(x)]$ ，首先需要从p分布中采样，再通过 $\frac{1}{N} \sum_{i=1}^{N}f(x^i)$ 来计算。如果无法直接采样，可以通过另一个分布q间接实现。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115205350081.png" alt="image-20240115205350081" style="zoom:80%;" />

这样做会带来一些问题，尽管其期望是相同的，但是二者的方差并不相同，通过下面的推导可以看出来：

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115210132307.png" alt="image-20240115210132307" style="zoom:80%;" />

其中：

$$
E_{x \sim q} \left[ \left( f(x)\frac{p(x)}{q(x)} \right)^2 \right] 
= \int {f^2(x) \frac{p(x)}{q(x)} \frac{p(x)}{q(x)} q(x) dx} 
= \int {f^2(x) \frac{p(x)}{q(x)} p(x) dx} 
= E_{x \sim p} \left[ f^2(x)\frac{p(x)}{q(x)} \right]
$$
从上面的推导的结果可以看出，二者的方差相差了一个系数 $\frac{p(x)}{q(x)}$ 。如果p分布和q分布相差过大，那么有可能出现问题。

如下图所示，如果采样的次数足够多，就没什么问题。如果采样的次数不够多，有可能多次采样到q分布的右侧（概率更大），这样算出来的 $f(x)$ 的期望就是正数，一旦采样到q分布的左侧，就会乘以一个很大的权重，用以抵消右侧的数据，如果采样的次数不够多，可能就无法采样到q分布的左侧，导致最终的期望计算有偏差。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115211125473.png" alt="image-20240115211125473" style="zoom:80%;" />

#### Off-policy 中的梯度

经过上述推导，我们就得出了 Off-policy 中的梯度更新公式，同时也就可以反向得出损失函数。

其中：

- 我们假设 $p_{\theta}(s_t)$ 与 $p_{\theta ^{\prime}}(s_t)$ 大致相同；
- $\nabla logp_{\theta}=\frac{\nabla p_{\theta}}{p_{\theta}}$ ，将原式替换之后，把 $\nabla$ 拿到前面去，就可以得到损失函数的形式。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115214545282.png" alt="image-20240115214545282" style="zoom:80%;" />

### Proximal Policy Optimization (PPO)

上文中指出，如果两个分布 $p_{\theta}$ 和 $p_{\theta ^{\prime}}$ 相差过大，将会出现稳定性问题，而 PPO 正是用于防止两个分布相差过大。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115224554600.png" alt="image-20240115224554600" style="zoom:80%;" />

其中，$KL(\theta,\theta ^{\prime})$ 相当于惩罚项，表示 $\theta$ 与 $\theta ^{\prime}$ 接近的程度，全称为 Kullback–Leibler divergence ，公式如下，其具体性质可参考[维基百科](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ) 。
$$
D_{KL}(P \Vert Q) = \sum_{x \in \mathcal{X}}P(x) log \left( \frac{P(x)}{Q(x)} \right)
$$
由[Gibbs不等式](https://zh.wikipedia.org/wiki/吉布斯不等式)可知，当且仅当 $P=Q$ 时， $D_{KL}(P\Vert Q)$ 取最小值为零。

> Gibbs 不等式说明：
>
> 若 $\sum_{i=1}^{n}p_i = \sum_{i=1}^{n}q_i = 1$ ，且 $p_i, q_i \in (0,1]$ ，则有： $-\sum_{i=1}^{n} p_i log p_i \leq -\sum_{i=1}^{n} p_i log q_i$，当且仅当 $p_i=q_i ~~\forall i$ 时等号成立。
>
> Gibbs 不等式等价于：
> $$
> 0 \geq \sum_{i=1}^{n}p_i log q_i - \sum_{i=1}^{n}p_i log p_i = \sum_{i=1}^{n}p_i log(\frac{q_i}{p_i}) = -D_{KL}(P \Vert Q)
> $$
> 证明最右的项小于或等于0的方法有几种：
>
> - 已知 $ln(x) \leq x-1$ ，当且仅当 $x=1$ 时等号成立：
>   $$
>   \sum_{i=1}^{n} p_i log \frac{q_i}{p_i} \leq \sum_{i=1}^{n} p_i (\frac{q_i}{p_i}-1) = \sum_{i=1}^{n} (q_i - p_i) = \sum_{i=1}^{n} q_i - \sum_{i=1}^{n} p_i = 0
>   $$
>   
>- 根据[对数求和不等式](https://zh.wikipedia.org/wiki/%E5%AF%B9%E6%95%B0%E6%B1%82%E5%92%8C%E4%B8%8D%E7%AD%89%E5%BC%8F)或[Jensen's不等式](https://zh.wikipedia.org/wiki/%E5%BB%B6%E6%A3%AE%E4%B8%8D%E7%AD%89%E5%BC%8F)：
>   $$
>   \sum_{i}p_i log \frac{q_i}{p_i} \leq log \sum_{i}p_i \frac{q_i}{p_i} = log \sum_{i}q_i \leq 0
>   $$

$\beta$ 相当于惩罚系数，如果KL值比较大，那么可以适当减小 $\beta$ ；如果KL值比较小，那么可以适当增加 $\beta$ 。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115232446696.png" alt="image-20240115232446696" style="zoom:80%;" />

### PPO2

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115233603609.png" alt="image-20240115233603609" style="zoom:80%;" />

其中 $clip(a,b,c)$ 表示：若 $a<b$ ，则输出 $b$ ；若 $a>c$ ，则输出 $c$ ；否则输出 $a$ 。$\epsilon$ 取值范围为0-1。

下图中，横轴表示 $\frac{p_{\theta}(a_t|s_t)}{p_{\theta ^k}(a_t|s_t)}$ 。若 $A>0$ ，最大化回报，将使得 $p_{\theta}(a_t|s_t)$ 增大，但是最大不能偏离 $p_{\theta ^k}(a_t|s_t)$ 太多；若 $A<0$ ，最大化回报，将使得 $p_{\theta}(a_t|s_t)$ 减小，但是最小不能偏离 $p_{\theta ^k}(a_t|s_t)$ 太多。

![image-20240115234959217](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240115234959217.png)

------



## DPO 推导

参考：

https://zhuanlan.zhihu.com/p/676371444

https://www.youtube.com/watch?v=NLU2hIbIDbA

我们最终需要优化的结果如下：

![image-20240118220402469](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240118220402469.png)

下面将先阐述一般 RLHF 的流程，再推导 DPO 优化目标。

### 一般 RLHF 的流程

#### **Supervised fine-tuning (SFT)**

SFT的主要目的是使pre-train之后的模型学会以对话的形式返回结果，而不是去做单纯的文本生成，一般来说使用收集的高质量Instruction数据进行训练，是RLHF流程的起始步骤。

#### Reward Modeling (RM)

**Bradley–Terry model** 被用于对事物间的比较关系进行建模，比如说棋类比赛，棋手之间往往互有胜负，Bradley–Terry model可以根据这些胜负信息去为所有的职业棋手建模，为他们赋予一个内在的“分数”，通过这个分数可以一定程度上反应棋手的水平，并预测两个棋手之间对局的胜负概率。
$$
P(i>j)=\frac{e^{\beta_i}}{e^{\beta_i}+e^{\beta_j}}\\
=\frac{1}{1+e^{\beta_j - \beta_i}}=\frac{1}{1+e^{-(\beta_i - \beta_j)}}
$$
以上是模型对目标 $i$ 和 $j$ 的比较（或者胜负）关系的建模， $\beta$ 是 $i$ 和 $j$ 的内在分数，$P(i>j)$  是 $i$ 优于 $j$ 的概率。

在RLHF场景下，Bradley–Terry model可以用来对人类偏好进行建模（即用于训练**奖励模型 Reward Model** ，在深度强化学习中，奖励模型是一个神经网络）。 $\beta$ 是我们希望Reward Model对于每条样本计算出的模型返回的内在分数，而结果 $P$ 代表了人类偏好的概率。

在实际当中，我们会收集到标注员对样本间两两比较的结果，我们会使用极大似然估计去优化Reward Model，使RM得到的分数对样本间比较结果的预测最大程度上和训练数据保持一致。具体而言，不同的标注员对两个答案的优劣有不同的偏好，在收集大量标注数据之后，就就可以计算人类认为的 $P(i>j)$ 。

<img src="https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117122300849.png" alt="image-20240117122300849" style="zoom:80%;" />

求在数据集D上的对数似然，使之最大化就即可得到奖励模型的参数。

![image-20240117133453024](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117133453024.png)

#### RL Fine-Tuning

在求得奖励模型之后，为了使奖励最大化，同时不让微调之后的模型与原始模型相差太大，利用 PPO 算法求解。

![image-20240117133706845](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117133706845.png)

### DPO 推导

#### 推导 $\pi(y|x)$ 的显式表达

![image-20240117134729895](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117134729895.png)

![image-20240117135408730](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117135408730.png)

![image-20240117135424633](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117135424633.png)

这个显式表达依赖于reference model和reward model，因为Z无法把所有可能的y都算一遍，所以在实际当中无法计算。

#### 推导 $\pi(y|x)$ 的隐式表达

对于 $\pi(y|x)$ 的显式表达，两边取对数，简单变换可以得到：

![image-20240117183416931](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117183416931.png)

根据 BT model ：

![image-20240117183550556](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117183550556.png)

可以得到：

![image-20240117183711912](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117183711912.png)

其中 $\pi^*$ 与 $\pi$ 相等，因此我们可以通过这个概率等式来估计 $\pi$ 的参数。求负对数似然可以得到：

![image-20240117184239045](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117184239045.png)

极大化这个负对数似然可以得到 $\pi$ 的参数 $\theta$ 。

#### 推导负对数似然的梯度

对上述负对数似然求梯度可得：

![image-20240117184733920](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117184733920.png)

其中 $\hat{r}_{\theta}(x,y)=\beta log \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}$ 。

推导过程如下：

![image-20240117195215647](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117195215647.png)

直观上，DPO 梯度更新的每一步都在增加被人类偏好的数据的生成概率，并降低不被人类偏好的数据的生成概率。同时，这一个更新是加权的，模型的隐式 reward model 计算的 reward 误差越大，权重也越大。如果去除这个权重，模型效果会变差。

#### DPO 流程范式

在实际当中，DPO一般通过两步骤进行训练：

1. 首先用 $\pi_{ref}$ 生成样本对，由人类进行标注，得到人类偏好数据；
2. 在人类偏好数据上优化模型，最小化上述负对数似然。

但一般来说由人类去标注数据成本很高，而我们更容易去获得开源的偏好数据，步骤会调整为：

1. 得到 $\pi_{ref}$ ；
2. 在该 $\pi_{ref}$ 上，用人类偏好数据，通过DPO训练目标训练模型。

如果偏好数据是使用 $\pi^{SFT}$ 采样得到的，可直接令 $\pi_{ref}=\pi^{SFT}$ 。如果 $\pi^{SFT}$ 不可知，可以用如下式子，以减小真实参考分布$\pi^{SFT}$和 DPO 实际使用的 $\pi_{ref}$ 之间的分布偏移：

![image-20240117203954011](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240117203954011.png)

### DPO 理论分析

#### Your Language Model Is Secretly a Reward Model

**定义 1**. 对于给定的某个函数 $f$ ，我们认为两个奖励函数 $r(x,y)$ 和 $r^{\prime}(x,y)$ 当且仅当  $r(x,y) -r^{\prime}(x,y)=f(x)$  时是等价的。

**引理 1**. 在 Plackett-Luce 偏好框架下（特别是 Bradley-Terry）， 来自同一等价类的两个奖励函数具有相同的偏好分布。

> ![image-20240118131828194](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240118131828194.png)

**引理 2**. 来自同一等价类的两个奖励函数在有约束的强化学习问题下会导致相同的最优策略。

> ![image-20240118131902047](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240118131902047.png)

**定理 1**. 在温和的假设下，对于某个模型 $\pi(y|x)$ 和一个给定的参考模型 $\pi_{ref}(y|x)$ ，与 Plackett-Luce 模型（特别是 Bradley-Terry）一致的所有奖励类别都可以使用重参数化 $r(x,y)=\beta log\frac{\pi_{\theta}^{*}(y|x)}{\pi_{ref}^{*}(y|x)}$ 来表示。

> ![image-20240118143821490](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240118143821490.png)

根据引理2， $r$ 与 $r^{\prime}$ 是等价类，那么其得出来的策略是一样的，所以 $r$ 可以重参数化为 $r^{\prime}$ 。下面将证明同一个等价的奖励函数类中，奖励函数重参数化的结果是唯一的：

> ![image-20240118144312449](https://raw.githubusercontent.com/zhangyuanwang777/Picture-cloud/main/img/image-20240118144312449.png)

**直观理解：**

其实在上文推导过程中，我们就发现在Bradley–Terry model里，我们只关心两个样本间的差，具体样本的reward绝对值是多少是不重要的。如果给定 $x$ ，那么只要在RM计算下所有模型返回 $y$ 的奖励 $r$ 之间的差保持一致，那么RM就没有本质区别。而我们发现在上述RM的显式表达中，后面的 $logZ(x)$ 项与 $y$ 无关，也就是说给定 $x$ 的情况下只有 $\hat{r}_{\theta}(x,y)$ 需要关心。而在进行样本比较的时候两个样本的 $x$ 一定是一样的，所以我们完全可以用 $\hat{r}_{\theta}(x,y)$ 来作为我们RM的表示，也就是我们的语言模型隐含的RM。
$$
\hat{r}_{\theta}(x,y) = \beta log \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}
$$




