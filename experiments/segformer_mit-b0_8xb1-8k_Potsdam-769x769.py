_base_ = [
    '../configs/_base_/models/segformer_mit-b0.py',
    #'../configs/_base_/datasets/potsdam.py',
    '../configs/_base_/datasets/uda_isprs.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_20k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

## added by LYU: 2024/07/10
SAM_dict = dict(
    type='SAM',
    image_encoder_cfg=dict(
        type='mmpretrain.ViTSAM',
        arch=arch,
        img_size=1024,
        patch_size=16,
        out_channels=256,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
    ),
    prompt_encoder_cfg=dict(
        type='PromptEncoder',
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ),
    mask_decoder_cfg=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer=dict(
            type='TwoWayTransformer',
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
            ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ))

model = dict(
    type='EncoderDecoderwithSAM',
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
work_dir = './experiments/segformerb0_results/'
