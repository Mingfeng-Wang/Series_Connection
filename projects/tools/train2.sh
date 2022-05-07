#CUDA_LAUNCH_BLOCKING=1
python projects/tools/train.py projects/configs/sparse_rcnn_self/sparse_rcnn_r34_fpn_1x_coco_8stages_4bs_local_series_detach0stage_boxweights2.2_6_lr50_wd1.5.py --gpu-ids 1 #--resume-from  /home/wmf/Github/workdir/sparse_rcnn_r34_fpn_1x_coco_8stages_4bs_local_series_6reg_detach0stage_boxweights2.2_6_lr52.5/epoch_7.pth
