_base_ = './sparse_rcnn_r34_fpn_1x_coco_8stages_local.py'
num_proposals = 200
model = dict(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_proposals))
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2), type='GradientCumulativeOptimizerHook', cumulative_iters=2)
work_dir =  '/home/wmf/Github/workdir/sparse_rcnn_r34_fpn_1x_coco_200_8stages_local'
find_unused_parameters = True
checkpoint_config = dict(interval=1)
