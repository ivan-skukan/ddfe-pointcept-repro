_base_ = ["../_base_/default_runtime.py"]

batch_size = 2
mix_prob = 0.5
empty_cache = False
enable_amp = True
enable_wandb = False
num_worker = 2

grid_size = 0.20
epoch = 30
eval_epoch = 30
num_classes = 19
ignore_index = -1

ddfe_cfg = dict(
    out_channels=32,
    spherical_channels=16,
    displacement_channels=16,
    attention_hidden_channels=16,
    projection_shape=(512, 5120),
    projection_fov=(-30.0, 15.0),
    gaussian_sigmas=(10, 30, 50, 70),
    gaussian_truncate=3.0,
    normalize_gaussian=True,
    range_eps=1e-3,
    clip=dict(
        enabled=True,
        percentile_low=10.0,
        percentile_high=90.0,
        reservoir_size=1000,
        min_samples=16,
        eps=1e-6,
    ),
    sensors=dict(
        semantic_kitti=dict(id=0, hb=2048, vb=64, fov=(-24.8, 2.0)),
        nuscenes=dict(id=1, hb=1080, vb=32, fov=(-30.0, 10.0)),
        waymo=dict(id=2, hb=2560, vb=64, fov=(-17.6, 2.4)),
    ),
)

model = dict(
    type="DDFESegmentor",
    freeze_backbone=False,
    unfreeze_backbone_keywords=(),
    ddfe=ddfe_cfg,
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=32,
        num_classes=num_classes,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            weight=[
                3.1557,
                8.7029,
                7.8281,
                6.1354,
                6.3161,
                7.9937,
                8.9704,
                10.1922,
                1.6155,
                4.2187,
                1.9385,
                5.5455,
                2.0198,
                2.6261,
                1.3212,
                5.1102,
                2.5492,
                5.8585,
                7.3929,
            ],
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)

optimizer = dict(type="Adam", lr=1e-3)
scheduler = dict(type="ExpLR", gamma=0.99 ** epoch)

dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti"
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

train_transform = [
    dict(type="AddLidarSensor", sensor="semantic_kitti"),
    dict(type="BeamSubsample", sensor="semantic_kitti", keep="even", p=0.5),
    dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomRotate", angle=[0, 2], axis="z", center=[0, 0, 0], p=1.0),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomShift", shift=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_displacement=True,
    ),
    dict(type="Update", keys_dict={"grid_size": grid_size, "ddfe_mix3d": 1}),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "displacement",
            "segment",
            "lidar_sensor",
            "grid_size",
            "ddfe_mix3d",
        ),
        feat_keys=("coord",),
    ),
]

val_transform = [
    dict(type="AddLidarSensor", sensor="semantic_kitti"),
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
        return_displacement=True,
    ),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "displacement",
            "segment",
            "origin_segment",
            "inverse",
            "lidar_sensor",
        ),
        feat_keys=("coord",),
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=val_transform,
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="AddLidarSensor", sensor="semantic_kitti"),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size / 2,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_displacement=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "displacement", "index", "lidar_sensor"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
        ignore_index=ignore_index,
    ),
)

