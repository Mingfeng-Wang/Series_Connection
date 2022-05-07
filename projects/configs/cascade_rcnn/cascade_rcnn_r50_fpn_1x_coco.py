_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_4_local.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer_config = dict(_delete_=True, type='GradientCumulativeFp16OptimizerHook', cumulative_iters=4, loss_scale='dynamic')
work_dir =  '/home/wmf/Github/workdir/cascade_rcnn_r50_fpn_1x_coco_bs4'
