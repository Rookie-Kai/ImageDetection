# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/data/home/scv9243/run/mmdetection/checkpoints/resnet101-5d3b4d8f.pth')),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='/data/home/scv9243/run/mmdetection/data/balloon/train/',
        classes=classes,
        ann_file='/data/home/scv9243/run/mmdetection/data/balloon/train/annotation_coco.json'),
    val=dict(
        img_prefix='/data/home/scv9243/run/mmdetection/data/balloon/val/',
        classes=classes,
        ann_file='/data/home/scv9243/run/mmdetection/data/balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='/data/home/scv9243/run/mmdetection/data/balloon/val/',
        classes=classes,
        ann_file='/data/home/scv9243/run/mmdetection/data/balloon/val/annotation_coco.json'))

evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='segm_mAP')

optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001
)
optimizer_config = dict(grad_clip=None)
# 学习率策略
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# 运行器设置
runner = dict(
    type='EpochBasedRunner',
    max_epochs=100
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = '/data/home/scv9243/run/mmdetection/checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'