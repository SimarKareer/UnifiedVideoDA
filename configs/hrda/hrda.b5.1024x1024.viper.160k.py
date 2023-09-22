_base_ = ['./hrda.b0.1024x1024.viper.106k.py']

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes = 31
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook',
                init_kwargs={
                    'entity': "video-da",
                    'project': "viper-baseline"
                },
                interval=50,
                log_checkpoint=True,
                log_checkpoint_metadata=True,
                num_eval_images=100)
    ])