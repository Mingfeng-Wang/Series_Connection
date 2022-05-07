_base_ = './sparse_rcnn_r50_fpn_1x_coco_8bs_8stages_series_detach0.py'
model = dict(
    roi_head=dict(
        type='SparseSeriesRoIHeadUndetach0stage'))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeOptimizerHook', cumulative_iters=4)
work_dir =  '/home/wmf/Github/workdir/sparse_rcnn_r50_fpn_1x_coco_4bs_8stages_series_undetach0stage'
find_unused_parameters = True
checkpoint_config = dict(interval=2)
