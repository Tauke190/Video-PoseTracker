# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Linear, bias_init_with_prob, constant_init, normal_init,
                      build_activation_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads import AnchorFreeHead
import torch.distributions as distributions

from opera.core.bbox import build_assigner, build_sampler
from opera.core.keypoint import gaussian_radius, draw_umich_gaussian
from opera.models.utils import build_positional_encoding, build_transformer
from ..builder import HEADS, build_loss
from easydict import EasyDict
from collections import Counter

@HEADS.register_module()
class PAVEHeadMulFrames(AnchorFreeHead):
    """Head of `End-to-End Multi-Person Pose Estimation with Transformers`.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_kpt_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the keypoint regression head.
            Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): ConfigDict is used for
            building the Encoder and Decoder. Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_kpt (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_oks (obj:`mmcv.ConfigDict`|dict): Config of the
            regression oks loss. Default `OKSLoss`.
        loss_hm (obj:`mmcv.ConfigDict`|dict): Config of the
            regression heatmap loss. Default `NegLoss`.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        with_kpt_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to True.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_frames=3,
                 num_query=100,
                 num_kpt_fcs=2,
                 num_keypoints=17,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_kpt=dict(type='L1Loss', loss_weight=70.0),
                 loss_oks=dict(type='OKSLoss', loss_weight=2.0),
                 loss_hm=dict(type='CenterFocalLoss', loss_weight=4.0),
                 as_two_stage=True,
                 with_kpt_refine=True,
                 train_cfg=dict(
                     assigner=dict(
                         type='PoseHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         kpt_cost=dict(type='KptL1Cost', weight=70.0),
                         oks_cost=dict(type='OksCost', weight=7.0))),
                 loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=70.0),
                 loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=2.0),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            # assert 'assigner' in train_cfg, 'assigner should be provided '\
            #     'when train_cfg is set.'
            assigner = train_cfg['assigner']
            # assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            # assert loss_kpt['loss_weight'] == assigner['kpt_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_kpt_fcs = num_kpt_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.with_kpt_refine = with_kpt_refine
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')
        self.loss_cls = build_loss(loss_cls)
        self.loss_kpt = build_loss(loss_kpt)
        self.loss_kpt_rpn = build_loss(loss_kpt_rpn)
        self.loss_kpt_refine = build_loss(loss_kpt_refine)
        self.loss_oks = build_loss(loss_oks)
        self.loss_oks_refine = build_loss(loss_oks_refine)
        self.loss_hm = build_loss(loss_hm)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
            
        self._init_layers()
        
    def _init_layers(self):
        """Initialize classification branch and keypoint branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        kpt_branch = []
        kpt_branch.append(Linear(self.embed_dims, 512))
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            kpt_branch.append(Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(Linear(512, 2 * self.num_keypoints))
        kpt_branch = nn.Sequential(*kpt_branch)
        
        dec_fc_sigma_branch = []
        for _ in range(self.num_kpt_fcs):
            dec_fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
        dec_fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 2 * self.num_keypoints, norm=False))
        dec_fc_sigma_branch = nn.Sequential(*dec_fc_sigma_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last kpt_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        
        if self.num_frames == 5:
            self.pre_pre_kpt_branches = nn.ModuleList()
            for i in range(num_pred-1):
                kpt_branch = []
                kpt_branch.append(Linear(self.embed_dims, 512))
                kpt_branch.append(nn.ReLU())
                for _ in range(self.num_kpt_fcs):
                    kpt_branch.append(Linear(512, 512))
                    kpt_branch.append(nn.ReLU())
                kpt_branch.append(Linear(512, 2 * self.num_keypoints))
                kpt_branch = nn.Sequential(*kpt_branch)
                self.pre_pre_kpt_branches.append(kpt_branch)
        else:
            self.pre_pre_kpt_branches = None
        
        self.pre_kpt_branches = nn.ModuleList()
        for i in range(num_pred-1):
            kpt_branch = []
            kpt_branch.append(Linear(self.embed_dims, 512))
            kpt_branch.append(nn.ReLU())
            for _ in range(self.num_kpt_fcs):
                kpt_branch.append(Linear(512, 512))
                kpt_branch.append(nn.ReLU())
            kpt_branch.append(Linear(512, 2 * self.num_keypoints))
            kpt_branch = nn.Sequential(*kpt_branch)
            self.pre_kpt_branches.append(kpt_branch)
        
            
        self.next_kpt_branches = nn.ModuleList()
        for i in range(num_pred-1):
            kpt_branch = []
            kpt_branch.append(Linear(self.embed_dims, 512))
            kpt_branch.append(nn.ReLU())
            for _ in range(self.num_kpt_fcs):
                kpt_branch.append(Linear(512, 512))
                kpt_branch.append(nn.ReLU())
            kpt_branch.append(Linear(512, 2 * self.num_keypoints))
            kpt_branch = nn.Sequential(*kpt_branch)
            self.next_kpt_branches.append(kpt_branch)
            
        if self.num_frames == 5:
            self.next_next_kpt_branches = nn.ModuleList()
            for i in range(num_pred-1):
                kpt_branch = []
                kpt_branch.append(Linear(self.embed_dims, 512))
                kpt_branch.append(nn.ReLU())
                for _ in range(self.num_kpt_fcs):
                    kpt_branch.append(Linear(512, 512))
                    kpt_branch.append(nn.ReLU())
                kpt_branch.append(Linear(512, 2 * self.num_keypoints))
                kpt_branch = nn.Sequential(*kpt_branch)
                self.next_next_kpt_branches.append(kpt_branch)
        else:
            self.next_next_kpt_branches = None
        # -------------------------------------------------------

        if self.with_kpt_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.kpt_branches = _get_clones(kpt_branch, num_pred)
            self.dec_fc_sigma_branches = _get_clones(dec_fc_sigma_branch, num_pred)
            
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.kpt_branches = nn.ModuleList(
                [kpt_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query,
                                            self.embed_dims * 2)
        
        refine_kpt_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
            refine_kpt_branch.append(nn.ReLU())
        refine_kpt_branch.append(Linear(self.embed_dims, 2))
        refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
    
        refine_fc_sigma_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_fc_sigma_branch.append(Linear(self.embed_dims, self.embed_dims))
        refine_fc_sigma_branch.append(Linear_with_norm(self.embed_dims, 2, norm=False))
        refine_fc_sigma_branch = nn.Sequential(*refine_fc_sigma_branch)
        
        if self.with_kpt_refine:
            num_pred = self.transformer.refine_decoder.num_layers
            if self.num_frames == 5:
                self.pre_pre_refine_kpt_branches = nn.ModuleList()
                for i in range(num_pred):
                    refine_kpt_branch = []
                    for _ in range(self.num_kpt_fcs):
                        refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
                        refine_kpt_branch.append(nn.ReLU())
                    refine_kpt_branch.append(Linear(self.embed_dims, 2))
                    refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
                    self.pre_pre_refine_kpt_branches.append(refine_kpt_branch)
            else:
                self.pre_pre_refine_kpt_branches = None
                
            self.pre_refine_kpt_branches = nn.ModuleList()
            for i in range(num_pred):
                refine_kpt_branch = []
                for _ in range(self.num_kpt_fcs):
                    refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
                    refine_kpt_branch.append(nn.ReLU())
                refine_kpt_branch.append(Linear(self.embed_dims, 2))
                refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
                self.pre_refine_kpt_branches.append(refine_kpt_branch)
                
            self.next_refine_kpt_branches = nn.ModuleList()
            for i in range(num_pred):
                refine_kpt_branch = []
                for _ in range(self.num_kpt_fcs):
                    refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
                    refine_kpt_branch.append(nn.ReLU())
                refine_kpt_branch.append(Linear(self.embed_dims, 2))
                refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
                self.next_refine_kpt_branches.append(refine_kpt_branch)
            if self.num_frames == 5:
                self.next_next_refine_kpt_branches = nn.ModuleList()
                for i in range(num_pred):
                    refine_kpt_branch = []
                    for _ in range(self.num_kpt_fcs):
                        refine_kpt_branch.append(Linear(self.embed_dims, self.embed_dims))
                        refine_kpt_branch.append(nn.ReLU())
                    refine_kpt_branch.append(Linear(self.embed_dims, 2))
                    refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
                    self.next_next_refine_kpt_branches.append(refine_kpt_branch)
            else:
                self.next_next_refine_kpt_branches = None
            # -------------------------------------------------------
            self.refine_kpt_branches = _get_clones(refine_kpt_branch, num_pred)
            self.refine_fc_sigma_branches = _get_clones(refine_fc_sigma_branch, num_pred)
        self.fc_hm = Linear(self.embed_dims, self.num_keypoints)
        
        # add decoder flow model
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        enc_prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.enc_flow = RealNVP(nets, nett, masks, enc_prior)
        
        dec_prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.dec_flow = RealNVP(nets, nett, masks, dec_prior)
        
        prior = distributions.MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.flow = RealNVP(nets, nett, masks, prior)
        
    
    def init_weights(self):
        """Initialize weights of the PETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.kpt_branches:
            constant_init(m[-1], 0, bias=0)
        # for m in self.pre_kpt_branches:
        #     constant_init(m[-1], 0, bias=0)
        # for m in self.next_kpt_branches:
        #     constant_init(m[-1], 0, bias=0)
        # initialization of keypoint refinement branch
        if self.with_kpt_refine:
            for m in self.pre_refine_kpt_branches:
                constant_init(m[-1], 0, bias=0)
            # for m in self.refine_kpt_branches:
            #     constant_init(m[-1], 0, bias=0)
            # for m in self.next_refine_kpt_branches:
            #     constant_init(m[-1], 0, bias=0)
        # initialize bias for heatmap prediction
        bias_init = bias_init_with_prob(0.1)
        normal_init(self.fc_hm, std=0.01, bias=bias_init)

    def forward(self, mlvl_feats, img_metas, track_queries=None,
                track_reference_points=None, track_query_pos=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.
            track_queries (Tensor, optional): Track queries from previous window.
                Shape [bs//3, N_track, 256].
            track_reference_points (Tensor, optional): Reference points for track 
                queries. Shape [bs//3, N_track, 30].
            track_query_pos (Tensor, optional): Positional embeddings for track 
                queries. Shape [bs//3, N_track, 256].

        Returns:
            outputs_classes (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            outputs_kpts (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, K*2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, h*w, num_class). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
            enc_outputs_kpt (Tensor): The proposal generate from the
                encode feature map, has shape (N, h*w, K*2). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id // self.num_frames]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = self.query_embedding.weight # shape: num_querys, embed_dims
        
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_kpt, enc_outputs_sigma, hm_proto, memory, n_track = \
                self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    pre_pre_kpt_branches=self.pre_pre_kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    pre_kpt_branches=self.pre_kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    kpt_branches=self.kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    next_kpt_branches=self.next_kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    next_next_kpt_branches=self.next_next_kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    sigma_branches=self.dec_fc_sigma_branches \
                        if self.as_two_stage else None,  # noqa:E501
                    track_queries=track_queries,
                    track_reference_points=track_reference_points,
                    track_query_pos=track_query_pos,
            )
                
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_kpts = []
        output_sigmas = []
        
        # num_q accounts for track queries (may differ from self.num_query)
        num_q = hs.shape[2]  # n_track + n_detect

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            if self.num_frames == 5:
                pre_pre_reference = reference[:, :num_q]
                pre_reference = reference[:, num_q:num_q*2]
                next_reference = reference[:, num_q*3:num_q*4]
                next_next_reference = reference[:, num_q*4:]
                
                pre_pre_reference = inverse_sigmoid(pre_pre_reference)
                pre_pre_tmp_kpt = self.pre_pre_kpt_branches[lvl](hs[lvl])
                pre_pre_pose_kpt = (pre_pre_tmp_kpt + pre_pre_reference).sigmoid()
                
                pre_reference = inverse_sigmoid(pre_reference)
                pre_tmp_kpt = self.pre_kpt_branches[lvl](hs[lvl])
                pre_pose_kpt = (pre_tmp_kpt + pre_reference).sigmoid()
                
                next_reference = inverse_sigmoid(next_reference)
                next_tmp_kpt = self.next_kpt_branches[lvl](hs[lvl])
                next_pose_kpt = (next_tmp_kpt + next_reference).sigmoid()
                
                next_next_reference = inverse_sigmoid(next_next_reference)
                next_next_tmp_kpt = self.next_kpt_branches[lvl](hs[lvl])
                next_next_pose_kpt = (next_next_tmp_kpt + next_next_reference).sigmoid()
                
                reference = reference[:, num_q*2:num_q*3]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp_kpt = self.kpt_branches[lvl](hs[lvl])
                sigma = self.dec_fc_sigma_branches[lvl](hs[lvl])
                output_sigma = sigma.sigmoid()
                assert reference.shape[-1] == self.num_keypoints * 2
                tmp_kpt += reference
                outputs_kpt = tmp_kpt.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_kpts.append(outputs_kpt)
                output_sigmas.append(output_sigma)
                
            else:
                pre_reference = reference[:, :num_q]
                next_reference = reference[:, num_q*2:]
                
                pre_reference = inverse_sigmoid(pre_reference)
                pre_tmp_kpt = self.pre_kpt_branches[lvl](hs[lvl])
                pre_pose_kpt = (pre_tmp_kpt + pre_reference).sigmoid()
                
                next_reference = inverse_sigmoid(next_reference)
                next_tmp_kpt = self.next_kpt_branches[lvl](hs[lvl])
                next_pose_kpt = (next_tmp_kpt + next_reference).sigmoid()
                
                reference = reference[:, num_q:num_q*2]
                reference = inverse_sigmoid(reference)
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp_kpt = self.kpt_branches[lvl](hs[lvl])
                sigma = self.dec_fc_sigma_branches[lvl](hs[lvl])
                output_sigma = sigma.sigmoid()
                assert reference.shape[-1] == self.num_keypoints * 2
                tmp_kpt += reference
                outputs_kpt = tmp_kpt.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_kpts.append(outputs_kpt)
                output_sigmas.append(output_sigma)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_kpts = torch.stack(outputs_kpts)
        output_sigmas = torch.stack(output_sigmas)

        # Store last decoder hidden states for track query extraction
        self._last_hs = hs[-1]  # [bs//3, num_q, 256] from last decoder layer

        if hm_proto is not None:
            # get heatmap prediction (training phase)
            hm_memory, hm_mask = hm_proto
            hm_pred = self.fc_hm(hm_memory)
            hm_proto = (hm_pred.permute(0, 3, 1, 2), hm_mask)

        if self.as_two_stage:
            if self.num_frames == 5:
                return outputs_classes, outputs_kpts, output_sigmas, \
                    enc_outputs_class, enc_outputs_kpt.sigmoid(), \
                    enc_outputs_sigma.sigmoid(), hm_proto, memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt, n_track
            else:
                return outputs_classes, outputs_kpts, output_sigmas, \
                    enc_outputs_class, enc_outputs_kpt.sigmoid(),  \
                    enc_outputs_sigma.sigmoid(), hm_proto, memory, mlvl_masks, None, pre_pose_kpt, next_pose_kpt, None, n_track
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')
        
    def forward_refine(self, memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt,
                       refine_targets, losses,
                       img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        kpt_preds, kpt_targets, area_targets, kpt_weights = refine_targets
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            pos_kpt_preds = torch.zeros_like(kpt_preds[:1])
            if self.num_frames == 5:
                pos_kpt_preds = torch.concat([pos_kpt_preds, pos_kpt_preds, pos_kpt_preds, pos_kpt_preds, pos_kpt_preds], dim=0)
            else:
                pos_kpt_preds = torch.concat([pos_kpt_preds, pos_kpt_preds, pos_kpt_preds], dim=0)
            pos_img_inds = kpt_preds.new_zeros([1], dtype=torch.int64)
        else:
            if self.training:
                if self.num_frames == 5:
                    pre_pre_pose_kpt = pre_pre_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
                    pre_pose_kpt = pre_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
                    next_pose_kpt = next_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
                    next_next_pose_kpt = next_next_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
                else:
                    pre_pose_kpt = pre_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
                    next_pose_kpt = next_pose_kpt.flatten(0, 1) # bs*num_querys, num_keypoints*2
            
            if self.num_frames == 5:
                pos_kpt_preds = torch.concat([pre_pre_pose_kpt[pos_inds], pre_pose_kpt[pos_inds], kpt_preds[pos_inds], next_pose_kpt[pos_inds], next_next_pose_kpt[pos_inds]], dim=0)
            else:
                pos_kpt_preds = torch.concat([pre_pose_kpt[pos_inds], kpt_preds[pos_inds], next_pose_kpt[pos_inds]], dim=0)
            
            # Compute which batch element each positive query belongs to
            num_q_per_img = kpt_preds.shape[0] // max(1, mlvl_masks[0].size(0) // self.num_frames)
            pos_img_inds = (pos_inds.nonzero() / num_q_per_img).squeeze(1).to(
                torch.int64)

        hs, init_reference, inter_references = self.transformer.forward_refine(
            mlvl_masks,
            memory.reshape(memory.size(0), -1, self.num_frames, memory.size(-1)),
            pos_kpt_preds.detach(),
            pos_img_inds,
            pre_pre_kpt_branches=self.pre_pre_refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
            pre_kpt_branches=self.pre_refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
            kpt_branches=self.refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
            next_kpt_branches=self.next_refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
            next_next_kpt_branches=self.next_next_refine_kpt_branches if self.with_kpt_refine else None,  # noqa:E501
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_kpts = []
        outputs_sigmas = []
        outputs_scores = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            num_gts = reference.shape[0] // self.num_frames
            if self.num_frames == 5:
                reference = reference[num_gts*2:num_gts*3]
                reference = inverse_sigmoid(reference)
                tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])
                tmp_sigma = self.refine_fc_sigma_branches[lvl](hs[lvl]).sigmoid()
                score = 1 - tmp_sigma
                score = torch.mean(score, dim=2, keepdim=True)
                outputs_score = score
                outputs_scores.append(outputs_score)
                assert reference.shape[-1] == 2
                tmp_kpt += reference
                outputs_kpt = tmp_kpt.sigmoid()
                outputs_kpts.append(outputs_kpt)
                outputs_sigma = tmp_sigma
                outputs_sigmas.append(outputs_sigma)
            else:
                reference = reference[num_gts:num_gts*2]
                reference = inverse_sigmoid(reference)
                tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])
                tmp_sigma = self.refine_fc_sigma_branches[lvl](hs[lvl]).sigmoid()
                score = 1 - tmp_sigma
                score = torch.mean(score, dim=2, keepdim=True)
                outputs_score = score
                outputs_scores.append(outputs_score)
                assert reference.shape[-1] == 2
                tmp_kpt += reference
                outputs_kpt = tmp_kpt.sigmoid()
                outputs_kpts.append(outputs_kpt)
                outputs_sigma = tmp_sigma
                outputs_sigmas.append(outputs_sigma)
            
        outputs_kpts = torch.stack(outputs_kpts)
        outputs_sigmas = torch.stack(outputs_sigmas)
        outputs_scores = torch.stack(outputs_scores)

        if not self.training:
            return outputs_kpts, outputs_scores, outputs_sigmas

        batch_size = mlvl_masks[0].size(0) // self.num_frames
        factors = []
        # Compute actual queries per image (may differ from self.num_query when track queries present)
        num_q_actual = kpt_preds.shape[0] // max(1, batch_size)
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            factor = mlvl_masks[0].new_tensor(
                [img_w, img_h, img_w, img_h],
                dtype=torch.float32).unsqueeze(0).repeat(num_q_actual, 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)

        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        num_total_pos = kpt_weights.new_tensor([outputs_kpts.size(1)])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        pos_kpt_weights = kpt_weights[pos_inds]
        pos_kpt_targets = kpt_targets[pos_inds]
        # pos_kpt_targets_scaled = pos_kpt_targets * factors
        # pos_areas = area_targets[pos_inds]
        # pos_valid = kpt_weights[pos_inds, 0::2]
        
        for i, (kpt_refine_preds, kpt_refine_sigma) in enumerate(zip(outputs_kpts, outputs_sigmas)):
            if pos_inds.sum() == 0:
                loss_kpt = loss_oks = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                # losses[f'd{i}.loss_oks_refine'] = loss_oks
                continue
            # kpt rle loss
            bs = kpt_refine_preds.size(0)
            # pos_refine_preds = kpt_refine_preds.reshape(
            #     bs, -1)   # shape: bs, 30
            all_gts = pos_kpt_targets.reshape(bs, -1, 2)
            gt_weights = pos_kpt_weights.reshape(bs, -1, 2)
            
            pred = kpt_refine_preds
            sigma = kpt_refine_sigma
            bar_mu = (pred - all_gts) / sigma
            
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(bs, -1, 1)
            nf_loss = torch.log(sigma) - log_phi 
            
            output = EasyDict(
                pred_jts=pred,
                sigma=sigma,
                nf_loss=nf_loss
            )
            labels = EasyDict(
                target_uv=all_gts,
                target_uv_weight=gt_weights
            )
            
            loss_kpt = self.loss_kpt_refine(output, labels, num_valid_kpt)
            losses[f'd{i}.loss_kpt_refine'] = loss_kpt
            # # oks loss
            # losses[f'd{i}.loss_kpt_refine'] = loss_kpt
            # pos_refine_preds_scaled = pos_refine_preds * factors
            # assert (pos_areas > 0).all()
            # loss_oks = self.loss_oks_refine(
            #     pos_refine_preds_scaled,
            #     pos_kpt_targets_scaled,
            #     pos_valid,
            #     pos_areas,
            #     avg_factor=num_total_pos)
            # losses[f'd{i}.loss_oks_refine'] = loss_oks
        return losses

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_areas=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_keypoints (list[Tensor]): Ground truth keypoints of the image,
                shape (num_gts, K*3).
            gt_areas (list[Tensor]): Ground truth mask areas of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        n_track = outs[-1]
        memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt = outs[-7:-1]
        outs = outs[:-7]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_keypoints, gt_areas, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, gt_areas,
                                  img_metas)
        losses_and_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses, refine_targets = losses_and_targets
        # get pose refinement loss
        losses = self.forward_refine(memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt,
                                     refine_targets, losses, img_metas)
        return losses

    def forward_train_sequential(self, x, img_metas, gt_bboxes, gt_labels,
                                  gt_keypoints, gt_areas, gt_track_ids,
                                  gt_bboxes_ignore=None, track_state=None):
        """Forward training for sequential windows with track query propagation.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information.
            gt_bboxes (list[Tensor]): GT bboxes per image.
            gt_labels (list[Tensor]): GT labels per image.
            gt_keypoints (list[Tensor]): GT keypoints per image.
            gt_areas (list[Tensor]): GT areas per image.
            gt_track_ids (list[Tensor]): GT track IDs per image.
            gt_bboxes_ignore: Ignored bboxes.
            track_state (tuple or None): (track_queries, track_refs, track_qpos,
                prev_track_gt_ids) from previous window.

        Returns:
            tuple: (losses dict, track_state for next window)
        """
        track_queries = track_ref_points = track_qpos = prev_track_gt_ids = None
        if track_state is not None:
            track_queries, track_ref_points, track_qpos, prev_track_gt_ids = track_state

        outs = self(x, img_metas,
                    track_queries=track_queries,
                    track_reference_points=track_ref_points,
                    track_query_pos=track_qpos)
        n_track = outs[-1]
        memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt = outs[-7:-1]
        outs_for_loss = outs[:-7]

        if gt_labels is None:
            loss_inputs = outs_for_loss + (gt_bboxes, gt_keypoints, gt_areas, img_metas)
        else:
            loss_inputs = outs_for_loss + (gt_bboxes, gt_labels, gt_keypoints, gt_areas,
                                           img_metas)
        losses_and_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
            n_track=n_track, prev_track_gt_ids=prev_track_gt_ids,
            gt_track_ids=gt_track_ids)
        losses, refine_targets = losses_and_targets
        # get pose refinement loss
        losses = self.forward_refine(memory, mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt,
                                     next_pose_kpt, next_next_pose_kpt,
                                     refine_targets, losses, img_metas)

        # Extract track queries for next window
        # Use the last decoder layer outputs
        all_cls_scores, all_kpt_preds = outs_for_loss[0], outs_for_loss[1]
        last_hs_cls = all_cls_scores[-1]  # [bs//3, num_q, 1]
        last_kpt_preds = all_kpt_preds[-1]  # [bs//3, num_q, K*2]

        new_track_state = self._extract_track_queries(
            last_hs_cls, last_kpt_preds, outs_for_loss,
            n_track, gt_track_ids, img_metas)

        return losses, new_track_state

    def _extract_track_queries(self, cls_scores, kpt_preds, outs_for_loss,
                                n_track, gt_track_ids, img_metas,
                                score_thr=0.5, max_track_queries=100):
        """Extract high-confidence queries for propagation to next window.

        Args:
            cls_scores (Tensor): [bs//3, num_q, 1] classification scores from last dec layer.
            kpt_preds (Tensor): [bs//3, num_q, K*2] keypoint predictions from last dec layer.
            outs_for_loss: All outputs for loss computation.
            n_track (int): Number of track queries in current pass.
            gt_track_ids (list[Tensor]): GT track IDs for matching.
            img_metas (list[dict]): Image meta information.
            score_thr (float): Confidence threshold for track query selection.
            max_track_queries (int): Maximum number of track queries.

        Returns:
            tuple or None: (track_queries, track_refs, track_qpos, track_gt_ids) or None.
        """
        if not self.training:
            return None

        # Use scores from last classification layer
        scores = cls_scores.sigmoid().squeeze(-1)  # [bs//3, num_q]
        bs_div3 = scores.shape[0]

        # For simplicity, operate on first batch element (bs//3 should be 1 during sequential)
        scores_0 = scores[0]  # [num_q]
        mask = scores_0 > score_thr
        if mask.sum() == 0:
            return None

        # Cap at max_track_queries
        if mask.sum() > max_track_queries:
            topk_vals, topk_inds = scores_0.topk(max_track_queries)
            mask = torch.zeros_like(scores_0, dtype=torch.bool)
            mask[topk_inds] = True

        selected_inds = mask.nonzero(as_tuple=True)[0]

        # Extract from the decoder output: use the classification branch input (hs)
        # We need the actual hidden states. Since we have cls_scores = cls_branches(hs),
        # we can't invert that. Instead, we construct track queries from the
        # kpt_preds (reference points) and use a learned embedding as query content.
        # A simpler approach: re-use the output embeddings from the current query_embedding
        # weighting by the cls_scores. But the plan suggests using decoder hidden states.
        # Since we don't directly have hs here, we'll store it during forward().

        # Use stored hs from the forward pass
        if not hasattr(self, '_last_hs') or self._last_hs is None:
            return None

        last_hs = self._last_hs  # [bs//3, num_q, 256]
        track_queries = last_hs[:, selected_inds, :]  # [bs//3, N_sel, 256]
        # Reference points from current-frame kpt predictions (already sigmoid)
        track_refs = kpt_preds[:, selected_inds, :]  # [bs//3, N_sel, K*2]
        # Use zeros for track_query_pos (will be combined with learnable content)
        track_qpos = torch.zeros_like(track_queries)

        # Determine GT track IDs for selected queries via Hungarian assignment
        # We need to know which GT each selected query was assigned to
        track_gt_ids_list = []
        if gt_track_ids is not None and hasattr(self, '_last_assigned_gt_inds'):
            assigned_gt_inds = self._last_assigned_gt_inds  # list of tensors per image
            for b in range(bs_div3):
                gt_tids = gt_track_ids[b] if b < len(gt_track_ids) else None
                if gt_tids is not None and b < len(assigned_gt_inds):
                    agi = assigned_gt_inds[b]  # [num_q]
                    sel_agi = agi[selected_inds]  # [N_sel]
                    tids = torch.full((len(selected_inds),), -1, dtype=torch.int64,
                                      device=scores.device)
                    for i, gi in enumerate(sel_agi):
                        if gi > 0 and (gi - 1) < len(gt_tids):
                            tids[i] = gt_tids[gi - 1]
                    track_gt_ids_list.append(tids)
                else:
                    track_gt_ids_list.append(
                        torch.full((len(selected_inds),), -1, dtype=torch.int64,
                                    device=scores.device))
        else:
            track_gt_ids_list.append(
                torch.full((len(selected_inds),), -1, dtype=torch.int64,
                            device=scores.device))

        prev_track_gt_ids = track_gt_ids_list[0]  # For single batch

        return (track_queries.detach(), track_refs.detach(),
                track_qpos.detach(), prev_track_gt_ids.detach())

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def loss(self,
             all_cls_scores,
             all_kpt_preds,
             all_sigma_preds,
             enc_cls_scores,
             enc_kpt_preds,
             enc_sigma_preds,
             enc_hm_proto,
             gt_bboxes_list,
             gt_labels_list,
             gt_keypoints_list,
             gt_areas_list,
             img_metas,
             gt_bboxes_ignore=None,
             n_track=0,
             prev_track_gt_ids=None,
             gt_track_ids=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map, has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                    ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_keypoints_list = [
            gt_keypoints_list for _ in range(num_dec_layers)
        ]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_n_track = [n_track for _ in range(num_dec_layers)]
        all_prev_track_gt_ids = [prev_track_gt_ids for _ in range(num_dec_layers)]
        all_gt_track_ids = [gt_track_ids for _ in range(num_dec_layers)]

        losses_cls, losses_kpt, losses_oks, kpt_preds_list, kpt_targets_list, \
            area_targets_list, kpt_weights_list, track_metrics_list = multi_apply(
                self.loss_single, all_cls_scores, all_kpt_preds, all_sigma_preds,
                all_gt_labels_list, all_gt_keypoints_list,
                all_gt_areas_list, img_metas_list,
                all_n_track, all_prev_track_gt_ids, all_gt_track_ids)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, enc_sigma_preds, binary_labels_list,
                    gt_keypoints_list, gt_areas_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_kpt'] = enc_losses_kpt

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        # loss_dict['loss_oks'] = losses_oks[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_kpt_i, loss_oks_i in zip(
                losses_cls[:-1], losses_kpt[:-1], losses_oks[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            # loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            num_dec_layer += 1

        # losses of heatmap generated from P3 feature map
        # hm_pred, hm_mask = enc_hm_proto
        # loss_hm = self.loss_heatmap(hm_pred, hm_mask, gt_keypoints_list,
        #                             gt_labels_list, gt_bboxes_list)
        # loss_dict['loss_hm'] = loss_hm

        # Track query diagnostic metrics (from last decoder layer, detached)
        tm = track_metrics_list[-1]
        loss_dict['trk_kpt_l1'] = tm['track_kpt_l1']
        loss_dict['det_kpt_l1'] = tm['detect_kpt_l1']
        loss_dict['n_trk'] = tm['n_track']
        loss_dict['n_trk_pos'] = tm['n_track_pos']
        loss_dict['n_det_pos'] = tm['n_detect_pos']
        id_total = tm['track_id_total'].item()
        loss_dict['trk_id_con'] = tm['track_id_consistent'] / max(id_total, 1.0)

        return loss_dict, (kpt_preds_list[-1], kpt_targets_list[-1],
                           area_targets_list[-1], kpt_weights_list[-1])

    def loss_heatmap(self, hm_pred, hm_mask, gt_keypoints, gt_labels,
                     gt_bboxes):
        assert hm_pred.shape[-2:] == hm_mask.shape[-2:]
        num_img, _, h, w = hm_pred.size()
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = hm_pred.new_zeros(hm_pred.shape)
        for i, (gt_label, gt_bbox, gt_keypoint) in enumerate(
                zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.size(0) == 0:
                continue
            gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1,
                                              3).clone()
            gt_keypoint[..., :2] /= 8
            assert gt_keypoint[..., 0].max() <= w  # new coordinate system
            assert gt_keypoint[..., 1].max() <= h  # new coordinate system
            gt_bbox /= 8
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            for j in range(gt_label.size(0)):
                # get heatmap radius
                kp_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]), min_overlap=0.9)),
                    min=0, max=3)
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = torch.floor(gt_kp)
                        draw_umich_gaussian(hm_target[i, k], gt_kp_int,
                                            kp_radius)
        # compute heatmap loss
        hm_pred = torch.clamp(
            hm_pred.sigmoid_(), min=1e-4, max=1 - 1e-4)  # refer to CenterNet
        loss_hm = self.loss_hm(hm_pred, hm_target, mask=~hm_mask.unsqueeze(1))
        return loss_hm

    def loss_single(self,
                    cls_scores,
                    kpt_preds,
                    sigma_preds,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    img_metas,
                    n_track=0,
                    prev_track_gt_ids=None,
                    gt_track_ids=None):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            n_track (int): Number of track queries.
            prev_track_gt_ids (Tensor or None): GT track IDs from previous window.
            gt_track_ids (list[Tensor] or None): GT track IDs for current window.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        # matchCost
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas,
                                           n_track=n_track,
                                           prev_track_gt_ids=prev_track_gt_ids,
                                           gt_track_ids=gt_track_ids)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)
        area_targets = torch.cat(area_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale keypoints
        factors = []
        for img_meta, kpt_pred in zip(img_metas, kpt_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = kpt_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0).repeat(
                                              kpt_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        sigmas_preds = sigma_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        # compute keypoints rle loss
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            kpt_preds_tmp = torch.zeros_like(kpt_preds[:1])
            loss_kpt = kpt_preds_tmp.sum() * 0
        else:
            kpt_preds_tmp = kpt_preds[pos_inds]
            kpt_targets_tmp = kpt_targets[pos_inds] # num_gt, 30
            kpt_weights_tmp = kpt_weights[pos_inds] # num_gt, 30
            sigmas_preds = sigmas_preds[pos_inds] # num_gt, 30
            num_gt = kpt_preds_tmp.size(0)
            all_gts = kpt_targets_tmp.reshape(num_gt, -1, 2)
            gt_weights = kpt_weights_tmp.reshape(num_gt, -1, 2)
            pred = kpt_preds_tmp.reshape(num_gt, -1, 2)
            sigma = sigmas_preds.reshape(num_gt, -1, 2)
            bar_mu = (pred - all_gts) / sigma
            log_phi = self.dec_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(num_gt, -1, 1)
            nf_loss = torch.log(sigma) - log_phi
            output = EasyDict(
                pred_jts=pred,
                sigma=sigma,
                nf_loss=nf_loss
            )
            labels = EasyDict(
                target_uv=all_gts,
                target_uv_weight=gt_weights
            )
            loss_kpt = self.loss_kpt(output, labels, num_valid_kpt)
        # keypoint oks loss
        pos_inds = kpt_weights.sum(-1) > 0
        factors = factors[pos_inds][:, :2].repeat(1, kpt_preds.shape[-1] // 2)
        pos_kpt_preds = kpt_preds[pos_inds] * factors
        pos_kpt_targets = kpt_targets[pos_inds] * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds, 0::2]
        if len(pos_areas) == 0:
            loss_oks = pos_kpt_preds.sum() * 0
        else:
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks(
                pos_kpt_preds,
                pos_kpt_targets,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)

        # ---- Track query diagnostic metrics (detached, logging only) ----
        track_metrics = {}
        device = kpt_preds.device
        num_query_per_img = kpt_preds.shape[0] // num_imgs
        if n_track > 0 and num_imgs > 0:
            # pos_inds is a flat boolean mask over [num_imgs * num_query]
            # For sequential training (num_imgs=1), query i < n_track is a track query
            pos_indices = pos_inds.nonzero(as_tuple=True)[0]  # absolute indices
            track_pos = pos_indices[pos_indices < n_track]
            detect_pos = pos_indices[pos_indices >= n_track]

            # Per-query L1 keypoint error (as proxy metric, detached)
            if len(track_pos) > 0:
                track_l1 = (kpt_preds[track_pos] - kpt_targets[track_pos]).abs()
                track_l1 = (track_l1 * kpt_weights[track_pos]).sum() / max(kpt_weights[track_pos].sum(), 1.0)
                track_metrics['track_kpt_l1'] = track_l1.detach()
            else:
                track_metrics['track_kpt_l1'] = torch.tensor(0.0, device=device)

            if len(detect_pos) > 0:
                detect_l1 = (kpt_preds[detect_pos] - kpt_targets[detect_pos]).abs()
                detect_l1 = (detect_l1 * kpt_weights[detect_pos]).sum() / max(kpt_weights[detect_pos].sum(), 1.0)
                track_metrics['detect_kpt_l1'] = detect_l1.detach()
            else:
                track_metrics['detect_kpt_l1'] = torch.tensor(0.0, device=device)

            track_metrics['n_track_pos'] = torch.tensor(float(len(track_pos)), device=device)
            track_metrics['n_detect_pos'] = torch.tensor(float(len(detect_pos)), device=device)
            track_metrics['n_track'] = torch.tensor(float(n_track), device=device)

            # Track ID consistency: how many track queries re-matched the same person
            id_consistent = 0
            id_total = 0
            if (prev_track_gt_ids is not None and gt_track_ids is not None
                    and hasattr(self, '_last_assigned_gt_inds_per_img')
                    and len(self._last_assigned_gt_inds_per_img) > 0):
                assigned = self._last_assigned_gt_inds_per_img[0]  # [num_query] for image 0
                gt_tids = gt_track_ids[0] if len(gt_track_ids) > 0 else None
                if gt_tids is not None:
                    for qi in range(min(n_track, len(assigned))):
                        prev_tid = prev_track_gt_ids[qi].item() if qi < len(prev_track_gt_ids) else -1
                        if prev_tid < 0:
                            continue
                        id_total += 1
                        gi = assigned[qi].item()  # 1-based GT index, 0 = background
                        if gi > 0 and (gi - 1) < len(gt_tids):
                            cur_tid = gt_tids[gi - 1].item()
                            if cur_tid == prev_tid:
                                id_consistent += 1
            track_metrics['track_id_consistent'] = torch.tensor(float(id_consistent), device=device)
            track_metrics['track_id_total'] = torch.tensor(float(id_total), device=device)
        else:
            # No track queries in this pass
            track_metrics['track_kpt_l1'] = torch.tensor(0.0, device=device)
            track_metrics['detect_kpt_l1'] = torch.tensor(0.0, device=device)
            track_metrics['n_track_pos'] = torch.tensor(0.0, device=device)
            track_metrics['n_detect_pos'] = torch.tensor(float(pos_inds.sum().item()), device=device)
            track_metrics['n_track'] = torch.tensor(0.0, device=device)
            track_metrics['track_id_consistent'] = torch.tensor(0.0, device=device)
            track_metrics['track_id_total'] = torch.tensor(0.0, device=device)

        return loss_cls, loss_kpt, loss_oks, kpt_preds, kpt_targets, \
            area_targets, kpt_weights, track_metrics

    def get_targets(self,
                    cls_scores_list,
                    kpt_preds_list,
                    gt_labels_list,
                    gt_keypoints_list,
                    gt_areas_list,
                    img_metas,
                    n_track=0,
                    prev_track_gt_ids=None,
                    gt_track_ids=None):
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (x_{i}, y_{i}) and shape [num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3).
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            n_track (int): Number of track queries.
            prev_track_gt_ids: GT track IDs from previous window.
            gt_track_ids (list[Tensor] or None): GT track IDs for current window.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all
                    images.
                - kpt_targets_list (list[Tensor]): Keypoint targets for all
                    images.
                - kpt_weights_list (list[Tensor]): Keypoint weights for all
                    images.
                - area_targets_list (list[Tensor]): area targets for all
                    images.
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        # Build per-image track info lists for multi_apply
        n_track_list = [n_track for _ in range(len(img_metas))]
        prev_track_gt_ids_list = [prev_track_gt_ids for _ in range(len(img_metas))]
        gt_track_ids_list = []
        for i in range(len(img_metas)):
            if gt_track_ids is not None and i < len(gt_track_ids):
                gt_track_ids_list.append(gt_track_ids[i])
            else:
                gt_track_ids_list.append(None)

        # Reset per-image tracking for this call
        self._last_assigned_gt_inds_per_img = []

        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, kpt_preds_list, 
             gt_labels_list, gt_keypoints_list, gt_areas_list, img_metas,
             n_track_list, prev_track_gt_ids_list, gt_track_ids_list)

        # Store assigned_gt_inds for track query extraction
        self._last_assigned_gt_inds = self._last_assigned_gt_inds_per_img

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, kpt_targets_list,
                kpt_weights_list, area_targets_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           kpt_pred,
                           gt_labels,
                           gt_keypoints,
                           gt_areas,
                           img_meta,
                           n_track=0,
                           prev_track_gt_ids=None,
                           gt_track_ids_single=None):
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (x_{i}, y_{i}) and
                shape [num_query, K*2].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_keypoints (Tensor): Ground truth keypoints for one image with
                shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas (Tensor): Ground truth mask areas for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            n_track (int): Number of track queries.
            prev_track_gt_ids: GT track IDs assigned to track queries from prev window.
            gt_track_ids_single (Tensor or None): GT track IDs for this image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - kpt_targets (Tensor): Keypoint targets of each image.
                - kpt_weights (Tensor): Keypoint weights of each image.
                - area_targets (Tensor): Area targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = kpt_pred.size(0)
        # assigner and sampler - pass track-aware params if assigner supports them
        from opera.core.bbox.assigners.hungarian_assigner import TrackAwarePoseHungarianAssigner
        if isinstance(self.assigner, TrackAwarePoseHungarianAssigner):
            assign_result = self.assigner.assign(
                cls_score, kpt_pred, gt_labels, gt_keypoints, gt_areas, img_meta,
                n_track=n_track, prev_track_gt_ids=prev_track_gt_ids,
                gt_track_ids=gt_track_ids_single)
        else:
            assign_result = self.assigner.assign(cls_score, kpt_pred, gt_labels,
                                                 gt_keypoints, gt_areas, img_meta)

        # Store assigned_gt_inds for track query extraction
        if not hasattr(self, '_last_assigned_gt_inds_per_img'):
            self._last_assigned_gt_inds_per_img = []
        self._last_assigned_gt_inds_per_img.append(assign_result.gt_inds.clone())

        sampling_result = self.sampler.sample(assign_result, kpt_pred,
                                              gt_keypoints)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_labels.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_bboxes)

        img_h, img_w, _ = img_meta['img_shape']

        # keypoint targets
        kpt_targets = torch.zeros_like(kpt_pred)
        kpt_weights = torch.zeros_like(kpt_pred)
        pos_gt_kpts = gt_keypoints[sampling_result.pos_assigned_gt_inds]
        pos_gt_kpts = pos_gt_kpts.reshape(pos_gt_kpts.shape[0],
                                          pos_gt_kpts.shape[-1] // 3, 3)
        valid_idx = pos_gt_kpts[:, :, 2] > 0
        pos_kpt_weights = kpt_weights[pos_inds].reshape(
            pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2)
        pos_kpt_weights[valid_idx] = 1.0
        kpt_weights[pos_inds] = pos_kpt_weights.reshape(
            pos_kpt_weights.shape[0], kpt_pred.shape[-1])

        factor = kpt_pred.new_tensor([img_w, img_h]).unsqueeze(0)
        pos_gt_kpts_normalized = pos_gt_kpts[..., :2]
        pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
            factor[:, 0:1]
        pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
            factor[:, 1:2]
        kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
            pos_gt_kpts.shape[0], kpt_pred.shape[-1])

        area_targets = kpt_pred.new_zeros(
            kpt_pred.shape[0])  # get areas for calculating oks
        pos_gt_areas = gt_areas[sampling_result.pos_assigned_gt_inds]
        area_targets[pos_inds] = pos_gt_areas

        return (labels, label_weights, kpt_targets, kpt_weights,
                area_targets, pos_inds, neg_inds)

    def loss_single_rpn(self,
                        cls_scores,
                        kpt_preds,
                        sigma_preds,
                        gt_labels_list,
                        gt_keypoints_list,
                        gt_areas_list,
                        img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        kpt_targets = torch.cat(kpt_targets_list, 0)
        kpt_weights = torch.cat(kpt_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        # num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.shape[-1])
        sigma_preds = sigma_preds.reshape(-1, kpt_preds.shape[-1])
        num_valid_kpt = torch.clamp(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        # compute keypoints rle loss
        pos_inds = kpt_weights.sum(-1) > 0
        if pos_inds.sum() == 0:
            kpt_preds_tmp = torch.zeros_like(kpt_preds[:1])
            loss_kpt = kpt_preds_tmp.sum() * 0
        else:
            kpt_preds_tmp = kpt_preds[pos_inds]
            kpt_targets_tmp = kpt_targets[pos_inds] # num_gt, 30
            kpt_weights_tmp = kpt_weights[pos_inds] # num_gt, 30
            sigma_preds = sigma_preds[pos_inds] # num_gt, 30
            
            num_gt = kpt_preds_tmp.size(0)
            all_gts = kpt_targets_tmp.reshape(num_gt, -1, 2)
            gt_weights = kpt_weights_tmp.reshape(num_gt, -1, 2)
            pred = kpt_preds_tmp.reshape(num_gt, -1, 2)
            sigma = sigma_preds.reshape(num_gt, -1, 2)
            bar_mu = (pred - all_gts) / sigma
            log_phi = self.enc_flow.log_prob(bar_mu.reshape(-1, 2)).reshape(num_gt, -1, 1)
            nf_loss = torch.log(sigma) - log_phi
            output = EasyDict(
                pred_jts=pred,
                sigma=sigma,
                nf_loss=nf_loss
            )
            labels = EasyDict(
                target_uv=all_gts,
                target_uv_weight=gt_weights
            )
            loss_kpt = self.loss_kpt_rpn(output, labels, num_valid_kpt)
        # num_valid_kpt = torch.clamp(
        #     reduce_mean(kpt_weights.sum()), min=1).item()
        # # assert num_valid_kpt == (kpt_targets>0).sum().item()
        # loss_kpt = self.loss_kpt_rpn(
        #     kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        return loss_cls, loss_kpt

    @force_fp32(apply_to=('all_cls_scores', 'all_kpt_preds'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_kpt_preds,
                   all_sigma_preds,
                   enc_cls_scores,
                   enc_kpt_preds,
                   enc_sigma_preds,
                   hm_proto,
                   memory,
                   mlvl_masks,
                   pre_pre_pose_kpt,
                   pre_pose_kpt,
                   next_pose_kpt,
                   next_next_pose_kpt,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is an (n, K, 3) tensor
                with [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        cls_scores = all_cls_scores[-1]
        kpt_preds = all_kpt_preds[-1]
        # cls_scores = enc_cls_scores
        # kpt_preds = enc_kpt_preds

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            kpt_pred = kpt_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # TODO: only support single image test
            # memory_i = memory[:, img_id, :]
            # mlvl_mask = mlvl_masks[img_id]
            proposals = self._get_bboxes_single(cls_score, kpt_pred, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt,
                                                img_shape, scale_factor,
                                                memory, mlvl_masks, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           kpt_pred,
                           pre_pre_pose_kpt,
                           pre_pose_kpt,
                           next_pose_kpt,
                           next_next_pose_kpt,
                           img_shape,
                           scale_factor,
                           memory,
                           mlvl_masks,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (x_{i}, y_{i}) and
                shape [num_query, K*2].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],
                    where the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with
                    shape [num_query].
                - det_kpts: Predicted keypoints with shape [num_query, K, 3].
        """
        assert len(cls_score) == len(kpt_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            kpt_pred = kpt_pred[bbox_index]
            if self.num_frames == 5:
                pre_pre_pose_kpt = pre_pre_pose_kpt.flatten(0, 1)[bbox_index]
                pre_pose_kpt = pre_pose_kpt.flatten(0, 1)[bbox_index]
                next_pose_kpt = next_pose_kpt.flatten(0, 1)[bbox_index]
                next_next_pose_kpt = next_next_pose_kpt.flatten(0, 1)[bbox_index]
            else:
                pre_pose_kpt = pre_pose_kpt.flatten(0, 1)[bbox_index]
                next_pose_kpt = next_pose_kpt.flatten(0, 1)[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            kpt_pred = kpt_pred[bbox_index]
            det_labels = det_labels[bbox_index]


        # ----- results after joint decoder (default) -----
        # import time
        # start = time.time()
        refine_targets = (kpt_pred, None, None, torch.ones_like(kpt_pred))
        refine_outputs = self.forward_refine(memory.reshape(memory.size(0), -1, self.num_frames, memory.size(-1)), mlvl_masks, pre_pre_pose_kpt, pre_pose_kpt, next_pose_kpt, next_next_pose_kpt,
                                                refine_targets, None, None)
        # end = time.time()
        # print(f'refine time: {end - start:.6f}')

        det_kpts = refine_outputs[0][-1]
        det_kpts_score = refine_outputs[1][-1] # shape： 100, 15, 1
        det_kpts_sigmas = refine_outputs[2][-1] # shape: 100, 15, 2

        det_kpts[..., 0] = det_kpts[..., 0] * img_shape[1]
        det_kpts[..., 1] = det_kpts[..., 1] * img_shape[0]
        det_kpts[..., 0].clamp_(min=0, max=img_shape[1])
        det_kpts[..., 1].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_kpts /= det_kpts.new_tensor(
                scale_factor[:2]).unsqueeze(0).unsqueeze(0)

        # use circumscribed rectangle box of keypoints as det bboxes
        x1 = det_kpts[..., 0].min(dim=1, keepdim=True)[0]
        y1 = det_kpts[..., 1].min(dim=1, keepdim=True)[0]
        x2 = det_kpts[..., 0].max(dim=1, keepdim=True)[0]
        y2 = det_kpts[..., 1].max(dim=1, keepdim=True)[0]
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        
        output_regression_sigma = det_kpts_sigmas
        output_regression = det_kpts
        
        output_regression_p = self.get_p(output_regression_sigma)
        p_to_coord_index = 5
        output_regression = (output_regression * output_regression_p**p_to_coord_index) / (output_regression_p**p_to_coord_index + 1e-10)

        output_regression_score = output_regression_p
    
        det_kpts_score = output_regression_score
        det_kpts = output_regression
        
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        
        kpt_scores = scores[:, None, None] * det_kpts_score

        det_kpts = torch.cat(
            (det_kpts, kpt_scores), dim=2)

        return det_bboxes, det_labels, det_kpts, bbox_index

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        # Remove n_track from outs before passing to get_bboxes
        outs_for_bboxes = outs[:-1]  # exclude n_track
        results_list = self.get_bboxes(*outs_for_bboxes, img_metas, rescale=rescale)
        return results_list
    
    def get_p(self, output_regression_sigma, p_x = 0.2):
        output_regression_p = (1- torch.exp(-(p_x/output_regression_sigma)))
        output_regression_p = output_regression_p[:,:,0] * output_regression_p[:,:,1]
        output_regression_p = output_regression_p[:,:,None]
        return output_regression_p * 0.7


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)

def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))



class Linear_with_norm(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear_with_norm, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y
    
def oks_nms(poses, scores, thresh, sigmas=None, in_vis_thre=None):
    poses = poses.cpu().numpy()
    scores = scores.cpu().numpy()
    if len(poses) == 0: return []
    areas = (np.max(poses[:, :, 0], axis=1) - np.min(poses[:, :, 0], axis=1)) * \
            (np.max(poses[:, :, 1], axis=1) - np.min(poses[:, :, 1], axis=1))
    poses = poses.reshape(poses.shape[0], -1)

    order = scores.argsort()[::-1]

    keep = []
    keep_ind = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = oks_iou(poses[i], poses[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
        inds = np.where(oks_ovr <= thresh)[0]
        nms_inds = np.where(oks_ovr > thresh)[0]
        nms_inds = order[nms_inds + 1]
        keep_ind.append(nms_inds.tolist())
        order = order[inds + 1]

    return keep, keep_ind

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

