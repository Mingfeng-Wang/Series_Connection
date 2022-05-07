import scipy.stats
import mmcv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot  as plt

from pycocotools.mask import iou
from mmcv import Config
from mmdet.datasets import build_dataset

cfg = Config.fromfile('/media/wmf/E453334932D7B3C31/workdir/sparse_rcnn_r34_fpn_1x_coco_8stages_local/sparse_rcnn_r34_fpn_1x_coco_8stages_local.py')
results = mmcv.load('/media/wmf/E453334932D7B3C31/workdir/sparse_rcnn_r34_fpn_1x_coco_8stages_local/test_results.pkl')
#ious_scores=mmcv.load("out.pkl")

dataset = build_dataset(cfg.data.test)

prog_bar = mmcv.ProgressBar(len(results))
ious_assigned=[]
scores_all=[]
ious_all_bbox=[]
#ious_assigned = ious_scores["ious"]
#scores_all = ious_scores["scores"]
for i, (result, ) in enumerate(zip(results)):
    
    data_info = dataset.prepare_train_img(i)
    anns =  data_info['ann_info'][0]
    
    gts_dict = defaultdict(list)
    for bb, label in zip(anns["bboxes"],anns["labels"]):
        gts_dict[label].append(bb)
    cats = set(anns['labels'])
    for cat in cats:
        gts = gts_dict[cat]
        iscrowd = [0] * len(gts)
        dets = result[cat]
        if dets.size == 0:
            continue
        ious = iou(dets[:,:4],gts,iscrowd)
        if not np.isfinite(ious).all():
            continue
        ious_assigned.extend(ious.max(0))
        ious_all_bbox.extend(ious.max(1))
        scores_all.extend(dets[ious.argmax(0),4])

    
    prog_bar.update()
iou_score = {"ious_assigned":ious_assigned, "scores":scores_all,"ious_all_bbox":ious_all_bbox}
pearson = scipy.stats.pearsonr(ious_assigned,scores_all)[0]
mmcv.dump(iou_score, 'vis/sparse_rcnn_r34_fpn_1x_coco_8stages_local.pkl')
plt.scatter(ious_assigned,scores_all,s=0.02)
plt.ylabel("Localization confidence") 
plt.xlabel("IoU with gt. Pearson: {}".format(pearson)) 
plt.savefig("vis/sparse_rcnn_r34_fpn_1x_coco_8stages_local.png") 
  
plt.show()

