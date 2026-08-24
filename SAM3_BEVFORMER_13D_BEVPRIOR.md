# SAM3 + BEVFormer 13D Query Init + BEV Prior 实现说明

## 1. 目标

这套实现的目标是把 SAM3 提供的 2D proposal 信息，注入到 camera-only 的 BEVFormer 3D 检测流程里，同时尽量保留 BEVFormer 原本的时空建模能力。

当前主线方法不是简单后处理，而是把 SAM3 信息接入模型内部的两个位置：

1. `SAM query init`
2. `BEV prior`

最终使用的方法可以概括为：

**SAM3 2D proposal -> 13D 几何 proposal -> Query 内容注入 + BEV query bias -> BEVFormer decoder 做 3D 检测**


## 2. 方法概览

### 2.1 原始 6D proposal

最早的 proposal 只有 6 维：

```text
[xc, yc, w, h, score, class_id]
```

这能告诉模型“图像上哪里像一个目标”，但不能告诉模型：

- proposal 来自哪个 camera
- 这个 proposal 对应的 3D 观察方向是什么
- 相机在 lidar/ego 坐标中的位置是什么

对多相机 3D 检测来说，这些几何信息很关键。

### 2.2 升级后的 13D proposal

当前主线把 proposal 扩展成 13 维：

```text
[xc, yc, w, h, score, class_id, cam_id,
 ray_dir_x, ray_dir_y, ray_dir_z,
 cam_pos_x, cam_pos_y, cam_pos_z]
```

其中：

- `xc, yc, w, h` 是相对原图归一化后的 2D 框几何
- `score` 是 SAM3 proposal 分数
- `class_id` 是映射到 nuScenes 10 类的类别编号
- `cam_id` 表示 proposal 来自哪一路相机
- `ray_dir_*` 是 2D 框中心点回投影到 lidar 坐标系后的单位射线方向
- `cam_pos_*` 是相机在 lidar 坐标系中的位置，做了缩放

这让模型拿到的信息从“有个 2D 框”升级成了“某个相机方向上，某个位置可能有这个类别的目标”。


## 3. 数据流

整体数据流如下：

```text
SAM3 检测 JSON
  -> LoadSAM3Proposals
  -> results["sam_proposals"], results["sam_proposal_mask"]
  -> CustomCollect3D 放入 img_metas
  -> BEVFormerHead._extract_sam_query_inputs()
  -> SAMProposalQueryEncoder 做 query init
  -> SAMBEVPriorEncoder 做 BEV prior
  -> BEVFormer transformer / decoder
  -> 3D bbox 预测
```

涉及的关键文件：

- `projects/mmdet3d_plugin/datasets/pipelines/loading.py`
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`
- `projects/mmdet3d_plugin/models/utils/sam_query_encoder.py`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
- `projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior.py`
- `tools/run_sam_bevprior_best.sh`


## 4. Proposal 加载与几何构造

文件：

- `projects/mmdet3d_plugin/datasets/pipelines/loading.py`

核心类：

- `LoadSAM3Proposals`

### 4.1 输入

`LoadSAM3Proposals` 从一个 SAM3 结果 JSON 中读取 `detections`，按 `sample_token` 和 camera 名称取出当前样本的 2D proposal。

配置入口示例：

```python
dict(
    type='LoadSAM3Proposals',
    sam3_results=sam3_results,
    max_proposals=50,
    min_score=0.30,
)
```

### 4.2 类别映射

proposal 的 `label` 会被映射成 nuScenes 10 类的 `class_id`：

- car
- truck
- construction_vehicle
- bus
- trailer
- barrier
- motorcycle
- bicycle
- pedestrian
- traffic_cone

如果 label 不在这个集合里，会直接过滤掉。

### 4.3 2D 几何特征

对于每个 2D box：

- `xc = (x1 + x2) / 2`
- `yc = (y1 + y2) / 2`
- `w = x2 - x1`
- `h = y2 - y1`

然后按当前 camera 图像宽高归一化：

```text
xc / width, yc / height, w / width, h / height
```

### 4.4 camera/ray 几何特征

`LoadSAM3Proposals._camera_geometry()` 会从 `results` 中拿：

- `cam_intrinsic` 或 `cam2img`
- `lidar2cam`

然后对 2D box 中心 `(x_px, y_px)` 做几何计算：

1. 用相机内参逆矩阵把像素点变成 camera ray
2. 用 `cam2lidar = inv(lidar2cam)` 把 ray 变到 lidar 坐标系
3. 对 ray 做单位化，得到 `ray_dir_xyz`
4. 从 `cam2lidar[:3, 3]` 取相机位置，得到 `cam_pos_xyz`

对应代码逻辑是：

```text
ray_cam = inv(K) * [x, y, 1]
ray_lidar = R_cam2lidar * ray_cam
ray_lidar = normalize(ray_lidar)
cam_pos = t_cam2lidar / cam_pos_scale
```

这里 `cam_pos_scale` 默认是 `100.0`，目的是让相机位置特征数值量级更稳定。

### 4.5 proposal 排序与截断

所有 camera 的 proposal 会：

1. 拼接到一起
2. 按 `score` 从高到低排序
3. 截断到 `max_proposals`

然后写入：

- `results["sam_proposals"]`
- `results["sam_proposal_mask"]`


## 5. 如何传进模型

文件：

- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`

关键点是 `CustomCollect3D` 的 `meta_keys` 里已经加入：

```python
'sam_proposals', 'sam_proposal_mask'
```

这意味着 dataloader 在组 batch 时，会把这些 proposal 放进 `img_metas`，后面 head 可以直接读取。


## 6. Query 初始化模块

文件：

- `projects/mmdet3d_plugin/models/utils/sam_query_encoder.py`

核心类：

- `SAMProposalQueryEncoder`

### 6.1 输入格式

这个编码器兼容三种 proposal：

- 6D: `[xc, yc, w, h, score, class_id]`
- 13D: 6D + `cam_id + ray_dir + cam_pos`
- 15D: 13D + `mask_box_fill + mask_area_frac`

当前主线训练实际用的是 **13D**。

### 6.2 编码方式

编码器会把不同来源的信息分开编码：

1. `geom = proposals[..., :5]`
   - 即 `xc, yc, w, h, score`
2. `class_embed`
   - 对 `class_id` 做 embedding
3. `camera_embed`
   - 对 `cam_id` 做 embedding
4. `ray_mlp`
   - 对 `ray_dir_xyz + cam_pos_xyz` 做 MLP 编码

最后把这些特征拼起来，送进主 MLP：

```text
[geom, mask_quality, class_feat, camera_feat, ray_feat] -> MLP -> query_delta
```

其中 `mask_quality` 在 13D 模式下会自动补零，不影响主线。

### 6.3 分数门控

输出的 `query_delta` 还会乘以 proposal 的 `score`：

```text
query_delta = query_delta * score
```

这相当于让高分 proposal 对 query 注入更强，低分 proposal 更弱。


## 7. 在 BEVFormerHead 中如何注入 query

文件：

- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

### 7.1 新增配置项

`BEVFormerHead` 增加了几组参数：

```python
use_sam_query_init
num_sam_queries
sam_query_scale
sam_query_hidden_dim
sam_cls_embed_dim
sam_num_cameras
sam_cam_embed_dim
sam_ray_embed_dim

use_sam_bev_prior
sam_bev_prior_scale
sam_bev_prior_radius
sam_bev_prior_hidden_dim
sam_cam_pos_scale
```

### 7.2 读入 batch proposal

`_extract_sam_query_inputs()` 会从每个样本的 `img_metas` 中读出 `sam_proposals`。

这里做了三件事：

1. 兼容 6D / 13D / 15D
2. 每个样本最多取 `num_sam_queries`
3. 对 batch 内不同长度 proposal 做 padding，得到：
   - `batch_props: [B, K, D]`
   - `batch_masks: [B, K]`

### 7.3 注入 object query

`_inject_sam_queries()` 的逻辑是：

1. 原始 `object_query_embeds` 形状是 `[num_query, 2 * embed_dims]`
2. 复制成 batch 形式 `[B, num_query, 2 * embed_dims]`
3. 用 `SAMProposalQueryEncoder` 把 proposal 编成 `sam_delta`
4. 把 `sam_delta` 加到前 `injected_slots` 个 object query 的后半部分

关键代码语义是：

```text
object_query_embeds[:, :K, embed_dims:] += sam_query_scale * sam_delta[:, :K]
```

也就是说：

- 当前实现只改 object query 的 content half
- 没改 reference point 初始化
- 也没改 positional half

所以它本质上是“带几何信息的 query 内容注入”。


## 8. BEV Prior 模块

文件：

- `projects/mmdet3d_plugin/models/utils/sam_query_encoder.py`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

核心类：

- `SAMBEVPriorEncoder`

### 8.1 设计动机

query init 只影响 decoder object query，本质上还是 proposal-level 提示。

BEV prior 的目标是把 SAM3 的 2D 几何信息，转成对整个 BEV 空间的软约束，让 encoder/decoder 在构建 BEV 表征前就拿到一点空间偏置。

### 8.2 BEV 网格中心

`SAMBEVPriorEncoder._grid_centers()` 会根据：

- `pc_range`
- `bev_h`
- `bev_w`

生成 BEV 网格中心点，形状是：

```text
[bev_h * bev_w, 3]
```

这里 z 维固定为 0。

### 8.3 如何根据 ray 生成 prior

对每个 proposal：

1. 取出 `rays = proposals[..., 7:10]`
2. 取出 `cam_pos = proposals[..., 10:13] * cam_pos_scale`
3. 对每个 BEV cell center，计算它到这条相机射线的最近距离
4. 用高斯形式把距离变成权重

核心思想是：

```text
离 proposal ray 越近的 BEV cell，prior 越大
```

具体上：

- `proj_len` 表示网格点在 ray 上的投影长度
- `closest` 表示网格点到 ray 的最近投影点
- `dist` 表示网格点到该 ray 的最短距离
- `weights = exp(-0.5 * (dist / radius)^2) * score`

然后对所有 proposal 在 proposal 维做 `max` 聚合：

```text
prior = max_k weights_k
```

这样得到每个样本的软 BEV prior。

### 8.4 prior 到 embedding 的映射

`prior` 本身只是一个标量场，形状接近：

```text
[B, bev_h * bev_w]
```

后面用一个小 MLP：

```text
Linear(1 -> hidden_dim) -> ReLU -> Linear(hidden_dim -> embed_dims)
```

把每个 BEV cell 的 prior 强度映射成和 `bev_queries` 同维度的 embedding delta。

### 8.5 注入 BEV query

在 `BEVFormerHead._inject_sam_bev_prior()` 里，最终做的是：

```text
bev_queries = bev_queries + sam_bev_prior_scale * prior_delta
```

注意这里返回的是 batch-wise 的 `bev_queries`，也就是每个样本可以有不同的 prior bias。


## 9. 主实验配置

基础配置文件：

- `projects/configs/bevformer/bevformer-tiny-sam3-query-13d-bevprior.py`

这个文件里定义了：

- `use_sam_query_init=True`
- `num_sam_queries=50`
- `sam_query_scale=1.0`
- `use_sam_bev_prior=True`
- `sam_bev_prior_scale=0.2`
- `sam_bev_prior_radius=4.0`

但是正式长训时，没有直接用这组默认值，而是通过启动脚本覆盖成了效果更好的保守设置。


## 10. 实际长训使用的参数

训练脚本：

- `tools/run_sam_bevprior_best.sh`

它在命令行里覆盖了关键参数：

```bash
model.pts_bbox_head.sam_bev_prior_scale=0.05
model.pts_bbox_head.sam_bev_prior_radius=2.0
```

所以这次长训的真实方法是：

**13D query init + BEV prior(scale=0.05, radius=2.0)**

另外，脚本还做了这些事：

- 激活 `bevformer` conda 环境
- 设置 `PYTHONPATH`
- `unset CC CXX CUDAHOSTCXX`
- 从原始 BEVFormer checkpoint 加载：

```text
work_dirs/bevformer_tiny_nuscenes_mini/epoch_24.pth
```

也就是说，这不是从零训练，而是：

1. 加载原始 BEVFormer 权重
2. 新加的 SAM 模块随机初始化
3. 在此基础上继续 finetune


## 11. 为什么这个实现有效

相对于 6D proposal，这个 13D + BEV prior 方案主要补了两类信息：

### 11.1 camera identity

同样的 2D 框中心 `(xc, yc)`，在不同 camera 里的物理含义完全不同。

加入 `cam_id` 之后，模型可以区分：

- `CAM_FRONT` 看到的前方目标
- `CAM_BACK_LEFT` 看到的后侧目标

### 11.2 3D ray geometry

加入 `ray_dir + cam_pos` 后，模型不只知道“有个框”，还知道：

- 这个框对应从哪里看出去
- 在 lidar/BEV 坐标里大致朝哪个方向

query init 利用的是 proposal 级别提示；BEV prior 利用的是空间级别提示。两者结合后，信息注入更完整。


## 12. 当前实现的边界

虽然这版已经比纯 6D 更强，但它还不是最终形态，主要有这些限制：

### 12.1 没有改 reference point 初始化

当前只改了 object query 的 content half，没有直接初始化 decoder reference points。

这意味着几何信息还是“提示”，不是显式 3D anchor。

### 12.2 BEV prior 还是启发式软约束

当前 prior 基于 ray 到 BEV cell 的距离做高斯衰减，没有显式深度，也没有类别特定几何约束。

所以 prior 太强时会伤 NDS，这也是前面调参时看到的现象。

### 12.3 多视角重复实例还没有做合并

现在多个 camera 的 proposal 是拼接后按分数排序，还没有做跨视角实例聚合。

这意味着同一个物体可能以多个 proposal 形式重复进入 query init。

### 12.4 mask 信息还不是主线

15D 的 `mask_box_fill` 和 `mask_area_frac` 已经有接口，但当前最佳主线不是靠 mask，而是靠几何 proposal + 保守 prior。


## 13. 当前实验结论

在已有实验里，主结论是：

1. **13D geometry query** 相比 6D 是有效的
2. **BEV prior** 在保守参数下可以进一步提升
3. prior 过强时会提高一些召回，但容易伤整体 NDS
4. 当前最佳主线是：

```text
13D query init + BEV prior(scale=0.05, radius=2)
```

长训结果中：

- 最终轮 `epoch_24`: `mAP 0.0171`, `NDS 0.0796`
- 最优 NDS checkpoint `epoch_13`: `mAP 0.0191`, `NDS 0.0851`
- 最优 mAP checkpoint `epoch_11`: `mAP 0.0236`, `NDS 0.0751`

如果以后继续做正式评估，优先建议使用：

- `epoch_13.pth`


## 14. 下一步怎么继续

如果继续沿这条线升级，最自然的方向有三个：

### 14.1 SAM3-guided dynamic query generator

在 `use_sam_query_init` 基础上继续升级：

- 按类别/尺度自适应 query 数
- 用 mask 中心、主方向、尺度先验初始化 query
- 对多相机重复实例做 BEV 聚合
- 增加 query dropout 防止过依赖 SAM3

### 14.2 reference point 级别几何初始化

把 proposal 不只注入 content half，还用于初始化 decoder reference points。

这样会更接近“用 2D 证据引导 3D 几何”的真正融合。

### 14.3 更稳的 BEV prior

可以尝试：

- 类别条件的 prior
- 近距/远距不同 radius
- 把 prior 从 query bias 变成 attention bias


## 15. 一句话总结

这套实现的核心，不是把 SAM3 当后处理，而是把它变成：

**带相机身份和 3D ray 几何的 proposal 先验，用来同时影响 BEVFormer 的 object query 和 BEV query。**

其中最有效的当前版本是：

**13D 几何 query init + 保守的 BEV prior(scale=0.05, radius=2)**。
