# dataset settings
dataset_type = 'UDA_Dataset_ISPRS'
data_root = 'data'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile_UDA'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        resize_type='Resize_UDA',
        scale=(769, 769),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

img_ratios = [0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #data_prefix=dict(
        #    img_path='Potsdam_IRRG_DA/img_dir/train', seg_map_path='Potsdam_IRRG_DA/ann_dir/train'),
        #ann_file='Potsdam_IRRG_DA/train.txt',
        #B_img_path='Vaihingen_IRRG_DA/img_dir/train',
        #B_img_file='Vaihingen_IRRG_DA/train.txt',
        #data_prefix=dict(
        #    img_path='Vaihingen_IRRG_DA/img_dir/train', seg_map_path='Vaihingen_IRRG_DA/ann_dir/train'),
        #ann_file='Vaihingen_IRRG_DA/train.txt',
        #B_img_path='Potsdam_IRRG_DA/img_dir/train',
        #B_img_file='Potsdam_IRRG_DA/train.txt',
        #data_prefix=dict(
        #    img_path='Vaihingen_IRRG_DA/img_dir/train', seg_map_path='Vaihingen_IRRG_DA/ann_dir/train'),
        #ann_file='Vaihingen_IRRG_DA/train.txt',
        #B_img_path='Potsdam_RGB_DA/img_dir/train',
        #B_img_file='Potsdam_RGB_DA/train.txt',
        data_prefix=dict(
            img_path='Potsdam_RGB_DA/img_dir/train', seg_map_path='Potsdam_RGB_DA/ann_dir/train'),
        ann_file='Potsdam_RGB_DA/train.txt',
        B_img_path='Vaihingen_IRRG_DA/img_dir/train',
        B_img_file='Vaihingen_IRRG_DA/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Vaihingen_IRRG_DA/img_dir/val', seg_map_path='Vaihingen_IRRG_DA/ann_dir/val'),
        ann_file='Vaihingen_IRRG_DA/val.txt',
        #data_prefix=dict(img_path='Potsdam_IRRG_DA/img_dir/val', seg_map_path='Potsdam_IRRG_DA/ann_dir/val'),
        #ann_file='Potsdam_IRRG_DA/val.txt',
        #data_prefix=dict(img_path='Potsdam_RGB_DA/img_dir/val', seg_map_path='Potsdam_RGB_DA/ann_dir/val'),
        #ann_file='Potsdam_RGB_DA/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
#val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore'])
test_evaluator = val_evaluator
