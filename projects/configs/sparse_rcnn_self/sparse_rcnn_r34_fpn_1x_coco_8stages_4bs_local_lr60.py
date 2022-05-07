_base_ = './sparse_rcnn_r34_fpn_1x_coco_8stages_local.py'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
#optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2,loss_scale='dynamic')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeOptimizerHook', cumulative_iters=4)
work_dir =  '/home/wmf/Github/workdir/sparse_rcnn_r34_fpn_1x_coco_8stages_4bs_local_lr60'
find_unused_parameters = True
checkpoint_config = dict(interval=1)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000060, weight_decay=0.0001)