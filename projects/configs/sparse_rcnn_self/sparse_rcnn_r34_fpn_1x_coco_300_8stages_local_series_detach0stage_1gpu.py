_base_ = './sparse_rcnn_r34_fpn_1x_coco_8stages_local_series_detach0stage.py'
num_proposals = 300
model = dict(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_proposals))
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeOptimizerHook', cumulative_iters=4)
work_dir =  '/home/wmf/Github/workdir/sparse_rcnn_r34_fpn_1x_coco_300_8stages_local_series_detach0stage2'
find_unused_parameters = True
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1
)