import torch
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import DIIHead


@HEADS.register_module()
class DIISeriesHead(DIIHead):

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, final):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))

        # instance interactive
        proposal_feat = proposal_feat.permute(1, 0,
                                              2).reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        if not final:
            reg_feat = obj_feat
            for reg_layer in self.reg_fcs:
                reg_feat = reg_layer(reg_feat)
            bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, -1)
        else:
            bbox_delta = None

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)

        return cls_score, bbox_delta, obj_feat.view(N, num_proposals, -1)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):
        """"Loss function of DIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses
