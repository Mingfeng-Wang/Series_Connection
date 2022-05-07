_base_ = './sparse_rcnn_r34_fpn_1x_coco_8stages_local_series_detach0stage.py'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
optimizer = dict(_delete_=True, type='AdamW', lr=0.000027, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeOptimizerHook', cumulative_iters=4)
work_dir =  '/home/wmf/Github/workdir/sparse_rcnn_r34_fpn_1x_coco_8stages_4bs_local_series_detach0stage_lr27'
find_unused_parameters = True
checkpoint_config = dict(interval=1)