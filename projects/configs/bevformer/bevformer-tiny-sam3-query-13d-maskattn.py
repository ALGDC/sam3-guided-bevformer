_base_ = ['./bevformer-tiny.py']

sam3_results = 'work_dirs/sam3_nuscenes_mini/sam3_2d_detections_with_masks.json'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

model = dict(
    pts_bbox_head=dict(
        use_sam_query_init=True,
        num_sam_queries=50,
        sam_query_scale=1.0,
        debug_sam_query=True,
        sam_ray_embed_dim=64,
        use_sam_bev_prior=True,
        sam_bev_prior_scale=0.05,
        sam_bev_prior_radius=2.0,
        transformer=dict(
            encoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            embed_dims=256,
                            use_sam_mask_attention=True,
                            sam_mask_attention_scale=0.5,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                        )
                    ],
                ),
            ),
        ),
    )
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadSAM3Proposals', sam3_results=sam3_results, max_proposals=50, min_score=0.30, proposal_dim=13),
    dict(type='LoadSAM3MaskPriors', sam3_results=sam3_results, min_score=0.30, prior_size=(32, 56), score_power=1.0, max_area_frac=0.60),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadSAM3Proposals', sam3_results=sam3_results, max_proposals=50, min_score=0.30, proposal_dim=13),
    dict(type='LoadSAM3MaskPriors', sam3_results=sam3_results, min_score=0.30, prior_size=(32, 56), score_power=1.0, max_area_frac=0.60),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
