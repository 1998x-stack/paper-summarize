https://chat.openai.com/share/b2a135f5-410a-43ee-b664-69f8212579e1

---

# PPO

## **摘要**
我们提出了一种新的策略梯度方法族，用于强化学习，这些方法在通过与环境的交互采样数据和使用随机梯度上升优化“替代”目标函数之间交替进行。标准策略梯度方法对每个数据样本执行一次梯度更新，我们提出了一种新的目标函数，使得可以进行多个时期的小批量更新。这些新方法，我们称之为近端策略优化（PPO），具有信任区域策略优化（TRPO）的一些好处，但它们实现起来更简单，更通用，并且在样本复杂性（经验上）方面更好。我们的实验在一系列基准任务上测试了PPO，包括模拟机器人运动和Atari游戏玩法，我们展示了PPO在其他在线策略梯度方法中的表现，并且总体上在样本复杂性、简单性和墙时间之间取得了良好的平衡。

## **1 引言**
近年来，已经提出了几种不同的方法用于带有神经网络函数逼近器的强化学习。主要竞争者是深度Q学习、"vanilla"策略梯度方法和信任区域/自然策略梯度方法。然而，在开发一种可扩展（到大型模型和并行实现）、数据高效且鲁棒的方法（即，在多种问题上无需调整超参数即可成功）方面还有改进的空间。Q学习（带有函数逼近）在许多简单问题上失败，并且理解不透彻；vanilla策略梯度方法的数据效率和鲁棒性较差；而信任区域策略优化（TRPO）相对复杂，并且与包括噪声（如dropout）或参数共享（在策略和价值函数之间，或与辅助任务）的架构不兼容。

本文旨在通过引入一种算法来改善当前状况，该算法在只使用一阶优化的情况下，实现了TRPO的数据效率和可靠性能。我们提出了一种带有剪辑概率比率的新目标，形成了策略性能的悲观估计（即，下限）。为了优化策略，我们在从策略采样数据和对采样数据进行几个时期的优化之间交替进行。我们的实验比较了各种不同版本的替代目标的性能，并发现带有剪辑概率比率的版本表现最佳。我们还将PPO与文献中的几种先前算法进行了比较。在连续控制任务上，它的性能优于我们比较的算法。在Atari上，它在样本复杂性方面表现显著优于A2C，并且与ACER相似，尽管它更简单。

接下来，我将继续提取并总结下一页的内容。

## **2 背景：策略优化**

### **2.1 策略梯度方法**
策略梯度方法通过计算策略梯度的估计器并将其插入随机梯度上升算法来工作。最常用的梯度估计器形式为
$$\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$$
其中，$\pi_\theta$ 是一个随机策略，$\hat{A}_t$ 是时间步t的优势函数的估计器。这里，期望 $\hat{\mathbb{E}}_t[...]$ 表示在交替采样和优化的算法中，有限批次样本的经验平均。使用自动微分软件的实现通过构建一个目标函数来工作，其梯度是策略梯度估计器；通过对目标 $L_{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$ 进行微分来获得估计器 $\hat{g}$。虽然在相同轨迹上对这个损失 $L_{PG}$ 执行多步优化很有吸引力，但这样做并不合理，实证上它经常导致破坏性的大规模策略更新。

### **2.2 信任区域方法**
在TRPO中，一个目标函数（“替代”目标）在策略更新大小的约束下被最大化。具体来说，
$$
\underset{\theta}{\text{maximize}} \hat{\mathbb{E}}t \left[ \frac{\pi\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]
$$
受制于 $\hat{\mathbb{E}}_t[KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \leq \delta$。这里，$\theta_{old}$ 是更新前的策略参数向量。这个问题可以通过使用共轭梯度算法在对目标进行线性近似和对约束进行二次近似后被高效近似解决。

TRPO的理论实际上建议使用惩罚而不是约束，即解决无约束优化问题
$$
\max_{\theta} \hat{\mathbb{E}}t \left[ \frac{\pi{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}t - \beta \text{KL}[\pi{\theta_{\text{old}}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)] \right]
$$
对于某些系数 $\beta$。这是因为某个替代目标（计算状态的最大KL而不是平均）形成了策略 $\pi$ 性能的下界（即，悲观界限）。TRPO使用硬约束而不是惩罚，因为很难选择一个在不同问题中表现良好的 $\beta$ 单一值——甚至在单一问题中，其中特征在学习过程中会发生变化。因此，为了实现我们的目标，即模仿TRPO的单调改进，实验表明，简单地选择一个固定的惩罚系数 $\beta$ 并用SGD优化惩罚目标方程（5）是不够的；还需要额外的修改。

接下来，我将继续提取并总结下一页的内容。

## **3 裁剪替代目标**

让 $r_t(\theta)$ 表示概率比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，所以 $r(\theta_{old}) = 1$。TRPO最大化一个“替代”目标
$$L_{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]$$
CPI的上标指的是保守的策略迭代，这个目标被提出。没有约束，$L_{CPI}$ 的最大化会导致过大的策略更新；因此，我们现在考虑如何修改目标，以惩罚将 $r_t(\theta)$ 从1移开的策略变化。

我们提出的主要目标是以下内容：
$$L_{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$
其中 $\epsilon$ 是一个超参数，比如 $\epsilon = 0.2$。这个目标的动机如下。内部的第一项是 $L_{CPI}$。第二项，$\text{clip}( r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t$，通过裁剪概率比率修改替代目标，这消除了将 $r_t$ 移出区间 $[1-\epsilon, 1+\epsilon]$ 的动机。最后，我们取裁剪和未裁剪目标的最小值，所以最终目标是未裁剪目标的下界（即，悲观界限）。有了这个方案，我们只在它会使目标改善时忽略概率比率的变化，并在它使目标变差时包括它。注意 $L_{CLIP}(\theta) = L_{CPI}(\theta)$ 在 $\theta_{old}$ 周围的一阶（即，$r=1$），然而，当 $\theta$ 从 $\theta_{old}$ 移开时，它们变得不同。图1绘制了 $L_{CLIP}$ 中的单个项（即，单个 $t$），注意概率比率 $r$ 根据优势是正还是负在 $1-\epsilon$ 或 $1+\epsilon$ 处被裁剪。

![LCLIP图示](https://ar5iv.labs.arxiv.org/html/1707.06347/assets/x1.png)

**图1**: 展示了替代函数 $L_{CLIP}$ 作为概率比率 $r$ 的函数的一个项（即，单个时间步）的图，对于正优势（左）和负优势（右）。每个图上的红圈显示了优化的起点，即 $r=1$。注意 $L_{CLIP}$ 总结了许多这样的项。

图2提供了关于替代目标 $L_{CLIP}$ 的另一个直觉来源。它展示了我们在连续控制问题上通过近端策略优化（我们将很快介绍的算法）沿策略更新方向插值时，几个目标是如何变化的。我们可以看到 $L_{CLIP}$ 是 $L_{CPI}$ 的下界，并且对于过大的策略更新有惩罚。

接下来，我将继续提取并总结下一页的内容。

## **4 自适应KL惩罚系数**

另一种方法，可以作为裁剪替代目标的替代方法，或者与之结合使用，是对KL散度使用惩罚，并适应惩罚系数，以便我们在每次策略更新时达到某个目标KL散度 $d_{\text{target}}$。在我们的实验中，我们发现KL惩罚的性能不如裁剪替代目标，然而，我们在这里包括它，因为它是一个重要的基线。

在这个算法的最简单实例中，我们在每次策略更新中执行以下步骤：
- 使用几个时期的小批量SGD，优化KL惩罚目标
  $$L_{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t - \beta KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)] \right]$$
- 计算 $d = \hat{\mathbb{E}}_t[KL[\pi_{\theta_{old}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]]$
  - 如果 $d < d_{\text{target}}/1.5$，则 $\beta \leftarrow \beta/2$
  - 如果 $d > 1.5 \cdot d_{\text{target}}$，则 $\beta \leftarrow 2 \cdot \beta$

下一次策略更新将使用更新后的 $\beta$。有了这个方案，我们偶尔会看到KL散度与 $d_{\text{target}}$ 显著不同的策略更新，然而，这些情况很少见，并且 $\beta$ 会迅速调整。上面的1.5和2是启发式选择的，但算法对它们并不非常敏感。初始的 $\beta$ 是另一个超参数，但在实践中并不重要，因为算法会迅速调整它。

## **5 算法**

前几节中的替代损失可以通过对典型策略梯度实现进行微小改变来计算和微分。对于使用自动微分的实现，只需构建损失 $L_{CLIP}$ 或 $L_{KLPEN}$ 而不是 $L_{PG}$，并在这个目标上执行多步随机梯度上升。

大多数用于计算方差减少的优势函数估计器的技术都使用了学习的状态值函数 $V(s)$；例如，广义优势估计或其他方法。

接下来，我将继续提取并总结下一页的内容。

## **5 算法（续）**

如果使用共享策略和价值函数参数的神经网络架构，我们必须使用一个结合了策略替代和价值函数误差项的损失函数。这个目标可以通过添加熵奖励来进一步增强，以确保足够的探索，正如过去的工作所建议的那样。结合这些项，我们得到以下目标，它（近似地）在每次迭代中被最大化：
$$L_{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_{CLIP}(\theta) - c_1 L_{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$
其中 $c_1, c_2$ 是系数，$S$ 表示熵奖励，$L_{VF}$ 是平方误差损失 $(V_\theta(s_t) - V_{\text{targ}}(s_t))^2$。

一种流行的策略梯度实现方式，特别适用于循环神经网络，运行策略 $T$ 时间步（其中 $T$ 远小于剧集长度），并使用收集的样本进行更新。这种方式需要一个不超过时间步 $T$ 的优势估计器。[Mni+16] 使用的估计器是
$$\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1} r_T - \gamma^{T-t} V(s_T)$$
其中 $t$ 指定了给定长度为 $T$ 的轨迹段内的时间索引。更一般地，我们可以使用截断的广义优势估计，当 $\lambda = 1$ 时，它简化为上述方程：
$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

使用固定长度轨迹段的近端策略优化（PPO）算法如下所示。每次迭代，每个 $N$（并行）执行者收集 $T$ 时间步的数据。然后我们在这些 $NT$ 时间步的数据上构建替代损失，并使用小批量SGD（或通常为了更好的性能，使用Adam）进行 $K$ 个时期的优化。

### **算法 1 PPO, Actor-Critic 风格**
对于迭代=1,2,...执行以下操作：
- 对于执行者=1,2,...,N：
  - 在环境中运行策略 $\pi_{\theta_{old}}$ $T$ 时间步
  - 计算优势估计 $\hat{A}_1,..., \hat{A}_T$
- 优化替代 $L$ 关于 $\theta$，使用 $K$ 个时期和小批量大小 $M \leq NT$
- $\theta_{old} \leftarrow \theta$

## **6 实验**

### **6.1 替代目标的比较**
首先，我们在不同的超参数下比较几种不同的替代目标。这里，我们将替代目标 $L_{CLIP}$ 与几种自然变体和删节版本进行比较。
- 无裁剪或惩罚：$L_t(\theta) = r_t(\theta) \hat{A}_t$
- 裁剪：$L_t(\theta) = \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)$
- KL惩罚（固定或自适应）：$L_t(\theta) = r_t(\theta) \hat{A}_t - \beta KL[\pi_{\theta_{old}}, \pi_\theta]$

接下来，我将继续提取并总结下一页的内容。

## **6 实验（续）**

### **6.1 替代目标的比较（续）**
对于KL惩罚，可以使用固定的惩罚系数 $\beta$ 或如第4节所述的自适应系数，并使用目标KL值 $d_{\text{target}}$。我们也尝试了在对数空间中进行裁剪，但发现性能并没有改善。

由于我们在搜索每种算法变体的超参数，我们选择了一个计算成本较低的基准来测试算法。具体来说，我们使用了OpenAI Gym中实现的7个模拟机器人任务，这些任务使用MuJoCo物理引擎。我们在每个任务上进行了一百万时间步的训练。除了用于裁剪（$\epsilon$）和KL惩罚（$\beta, d_{\text{target}}$）的超参数外，其他超参数在表3中给出。

为了表示策略，我们使用了一个带有两个隐藏层（每层64个单元）的全连接MLP，并使用tanh非线性函数，输出高斯分布的均值，标准差可变，遵循先前的工作。我们没有在策略和价值函数之间共享参数（所以系数 $c_1$ 无关紧要），我们也没有使用熵奖励。

每种算法在所有7个环境上运行，每个环境使用3个随机种子。我们通过计算最后100个剧集的平均总奖励来评分每次算法运行。我们对每个环境的分数进行了平移和缩放，使得随机策略得分为0，最佳结果设为1，并对21次运行取平均，为每种算法设置产生一个单一的标量。结果显示在表1中。注意，没有裁剪或惩罚的设置得分为负，因为对于一个环境（半猎豹），它导致了非常负的分数，比初始随机策略还要糟糕。

**表1**: 连续控制基准的结果。每种算法/超参数设置的平均归一化分数（在7个环境上的21次运行）。 $\beta$ 初始化为1。

### **6.2 与其他连续领域算法的比较**
接下来，我们将PPO（使用第3节中的“裁剪”替代目标）与文献中认为对连续问题有效的几种其他方法进行比较。我们与以下算法的调优实现进行了比较：信任区域策略优化、交叉熵方法（CEM）、带有自适应步长的vanilla策略梯度。

接下来，我将继续提取并总结下一页的内容。

## **6 实验（续）**

### **6.2 与其他连续领域算法的比较（续）**
A2C（Advantage Actor Critic）是A3C的同步版本，我们发现其性能与异步版本相同或更好。对于PPO，我们使用了前一节中的超参数，$\epsilon = 0.2$。我们看到PPO在几乎所有连续控制环境中都优于之前的方法。

**图3**: 在几个MuJoCo环境上，训练一百万时间步的几种算法比较。

### **6.3 连续领域展示：人形运动和转向**
为了展示PPO在高维连续控制问题上的性能，我们在一系列涉及3D人形机器人的问题上进行训练，其中机器人必须跑步、转向，并且可能在被立方体击中的同时从地面站起来。我们测试的三个任务是：(1) RoboschoolHumanoid：仅前进运动，(2) RoboschoolHumanoidFlagrun：目标位置每200时间步或每当达到目标时随机变化，(3) RoboschoolHumanoidFlagrunHarder，其中机器人被立方体击中并需要从地面站起来。图5展示了学习策略的静态帧，图4展示了三个任务的学习曲线。超参数在表4中给出。在同时进行的工作中，Heess等人使用PPO的自适应KL变体（第4节）来学习3D机器人的运动策略。

**图4**: 使用Roboschool在3D人形控制任务上的PPO学习曲线。

接下来，我将继续提取并总结下一页的内容。

## **6 实验（续）**

### **6.4 在Atari领域与其他算法的比较**
我们还在Arcade Learning Environment基准上运行了PPO，并与A2C和ACER的调优实现进行了比较。对于所有三种算法，我们使用了与[Mni+16]中相同的策略网络架构。PPO的超参数在表5中给出。对于其他两种算法，我们使用了为最大化此基准上的性能而调优的超参数。

所有49个游戏的结果表和学习曲线在附录B中提供。我们考虑以下两种评分指标：(1) 整个训练期间每个剧集的平均奖励（偏好快速学习），(2) 训练最后100个剧集的平均奖励（偏好最终性能）。表2显示了每种算法“赢得”的游戏数量，我们通过对三次试验的评分指标取平均来计算胜者。

**表2**: 每种算法“赢得”的游戏数量，评分指标在三次试验中取平均。

| 算法 | (1) 整个训练期间的平均剧集奖励 | (2) 最后100个剧集的平均剧集奖励 |
| --- | --- | --- |
| A2C | 1 | 1 |
| ACER | 18 | 28 |
| PPO | 30 | 19 |
| 平局 | 0 | 1 |

### **7 结论**
我们介绍了近端策略优化（PPO），这是一族策略优化方法，它们使用多个时期的随机梯度上升来执行每次策略更新。这些方法具有信任区域方法的稳定性和可靠性，但实现起来要简单得多，只需要对vanilla策略梯度实现进行少量代码更改，适用于更一般的设置（例如，当使用策略和价值函数的联合架构时），并且具有更好的整体性能。

### **8 致谢**
感谢OpenAI的Rocky Duan、Peter Chen和其他人提供的有见地的评论。

接下来，我将继续提取并总结下一页的内容。

The page primarily lists references cited throughout the paper. Here are a few key references with brief descriptions:

1. **[Bel+15] M. Bellemare et al.** - Discusses the Arcade Learning Environment, an evaluation platform for general agents, which is a significant benchmark in reinforcement learning.
2. **[Bro+16] G. Brockman et al.** - Introduces OpenAI Gym, a toolkit for developing and comparing reinforcement learning algorithms.
3. **[Dua+16] Y. Duan et al.** - Presents a benchmarking paper for deep reinforcement learning in continuous control.
4. **[Hee+17] N. Heess et al.** - Discusses the emergence of locomotion behaviors in rich environments, relevant to the continuous control tasks PPO is evaluated on.
5. **[KL02] S. Kakade and J. Langford** - Talks about approximately optimal approximate reinforcement learning, which is foundational for understanding policy optimization.
6. **[KB14] D. Kingma and J. Ba** - Introduces Adam, a method for stochastic optimization widely used in training deep learning models, including those in reinforcement learning.
7. **[Mni+15] V. Mnih et al.** - Describes human-level control through deep reinforcement learning, a seminal work in the field introducing the DQN algorithm.
8. **[Mni+16] V. Mnih et al.** - Discusses asynchronous methods for deep reinforcement learning, which are relevant for understanding the development of algorithms like A3C and A2C.
9. **[Sch+15a] J. Schulman et al.** - Introduces high-dimensional continuous control using generalized advantage estimation, relevant for understanding the advantage function used in PPO.
10. **[Sch+15b] J. Schulman et al.** - Introduces Trust Region Policy Optimization (TRPO), a precursor to PPO and a significant algorithm in policy optimization.
11. **[SL06] I. Szita and A. Lőrincz** - Discusses learning Tetris using the noisy cross-entropy method, relevant for understanding exploration strategies in reinforcement learning.
12. **[TET12] E. Todorov, T. Erez, and Y. Tassa** - Introduces MuJoCo, a physics engine for model-based control, used for the simulation environments in many reinforcement learning tasks.
13. **[Wan+16] Z. Wang et al.** - Discusses Sample Efficient Actor-Critic with Experience Replay, relevant for understanding sample efficiency in reinforcement learning.
14. **[Wil92] R. J. Williams** - Talks about simple statistical gradient-following algorithms for connectionist reinforcement learning, foundational for understanding policy gradient methods.

These references provide a context for the development and evaluation of Proximal Policy Optimization (PPO) and its place within the broader field of reinforcement learning.