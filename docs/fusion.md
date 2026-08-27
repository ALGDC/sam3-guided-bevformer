**Slide 1：为什么要融合 SAM3 和 BEVFormer？**

**Motivation**

BEVFormer 是一个 camera-only 3D 检测模型，可以从多摄像头图像中学习 BEV
表示。

但是 camera-only 3D detection 本身很难：

> 2D 图像没有直接的深度信息
>
> 小目标和远处目标很难定位
>
> BEV 空间特征学习需要更强的语义和空间提示

所以我们的想法是：

使用 SAM3 提供的伪标签作为额外监督，帮助 BEVFormer 在训练时学到更好的
BEV 表示。

**Slide 2：方法一：类别存在性监督**

**Semantic Presence Supervision**

第一个融合方式比较简单：

SAM3 告诉 BEVFormer：当前场景里可能有哪些类别。

例如 SAM3 在图像中检测到：

> car
>
> pedestrian
>
> traffic cone

那么我们希望 BEVFormer 的 decoder queries 中，至少有一些 query
对这些类别产生较高响应。

这个方法只使用：

SAM3 pseudo labels （因为是sam3推理的结果，不是ground
truth，所以我们称为伪标签）

不使用目标的 BEV 位置。

**Slide 3：类别存在性监督怎么做？**

**Query-level Class Response**

BEVFormer decoder 会输出很多 object queries。对于 SAM3
检测到的每一个类别，我们做一件事：

在所有 queries 中，

找到对该类别响应最高的 query，

然后鼓励这个响应变高。

如果 SAM3 认为图像中有 pedestrian，那么 BEVFormer 至少应该有一个 query
也认为这里存在 pedestrian。

$$L_{sem} = BCE\left( \max_{q}S_{q,c},1 \right)$$

$S_{q,c}$ 表示第 q 个 query 对类别 c 的分类 logit

$max_{q}$ 表示从所有 queries 中取最大响应

BCE: Binary Cross-Entropy

$$L = L_{BEVFormer} + \lambda_{sem}L_{sem}$$

$L_{BEVFormer}$：原始 BEVFormer detection loss

$\lambda_{sem}$：类别存在性监督的权重

$L_{sem}$：SAM3 提供的类别存在性辅助 loss

**Slide 4：实验结果**

**What does it help?**

类别存在性监督主要帮助模型小幅度提升：

> 类别感知能力
>
> 目标召回率
>
> 对小目标类别的响应

它的优点是简单、稳定。

但它也有一个限制：

它只告诉模型“有什么类别”，没有告诉模型“物体在哪里”。

**Slide 5：方法二：BEV 热图监督**

**BEV Heatmap Supervision**

这个方法使用三类 SAM3 信息：

> SAM3 pseudo labels → 类别
>
> SAM3 pseudo scores → 置信度
>
> SAM3 pseudo BEV indices → BEV 位置

因此它比前一个方法更强，因为它同时提供：

类别信息 + BEV 空间位置信息

**Slide 6：BEV 热图监督怎么做？**

**From SAM3 pseudo points to BEV heatmap**

BEVFormer encoder 会输出 BEV feature：*bev_embed*

我们在它后面加一个轻量的 heatmap head：*bev_embed → class-wise BEV
heatmap*

然后用 SAM3 的 pseudo BEV points 构建目标热图：

> pseudo label 决定类别通道
>
> pseudo BEV index 决定 BEV cell 位置
>
> pseudo score 决定监督强度

也就是说，对于一个 SAM3 伪目标：

> 类别是 car
>
> BEV index 是 (u, v)
>
> score 是 0.8

我们就在 car 这个类别的 BEV heatmap 上，把 (u,v) 附近作为正样本监督。

$$H = f_{hm}\left( F_{BEV} \right)$$

$F_{BEV}$：BEVFormer encoder 输出的 BEV feature

$f_{hm}$：轻量 heatmap head

H：预测得到的类别 BEV heatmap

$$T_{c,u,v} = p^{\gamma}$$

c 是 SAM3 pseudo label

(u, v) 是 SAM3 pseudo BEV index

p 是 SAM3 pseudo score

γ 用来调整置信度影响

然后使用加权 BCE：

$$L_{bev} = BCE(H,T)$$

最终 loss：

$$L = L_{BEVFormer} + \lambda_{bev}L_{bev}$$

因为 BEV heatmap
里绝大多数位置都是背景，所以我们还需要降低负样本位置的权重，避免 loss
被大量背景区域主导。

**Slide 7：方法优势总结**

**Advantages**

以上两种方法：

不直接把 SAM3 和 BEVFormer 的检测结果做后处理融合，而是把 SAM3
作为训练阶段的弱监督信号，引导 BEVFormer 学习更好的类别响应和 BEV
空间表示。
