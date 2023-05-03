# optimizer
optimizer = dict(
    type='SGD',
    lr=6e-5,
    momentum=0.9,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)
# optimizer_config = dict()
optimizer_config = None
