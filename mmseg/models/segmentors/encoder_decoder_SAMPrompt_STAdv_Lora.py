# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Dict, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

## added by LYU: 2024/04/26
from projects.sam_inference_demo import sam
## added by LYU: 2024/07/11
from mmengine.runner.checkpoint import load_checkpoint
import torch
## added by LYU: 2024/07/24
from .. import builder
from mmengine.optim import OptimWrapper
## added by LYU: 2024/07/26
from mmengine import MessageHub
from ..utils import resize
import numpy as np
import copy
## added by LYU: 2025/04/08
import math

model_zoo = {
    'base':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-base-p16_3rdparty_sa1b-1024x1024_20230413-78a25eed.pth',  # noqa
    'large':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-large-p16_3rdparty_sa1b-1024x1024_20230413-940520da.pth',  # noqa
    'huge':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-huge-p16_3rdparty_sa1b-1024x1024_20230413-faaf96f6.pth',  # noqa
}


@MODELS.register_module()
class EncoderDecoderwithSAMPromptSTAdvLora(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 SAM_arch: str = 'base',
                 SAM_config: OptConfigType = None,
                 discriminator_P: OptConfigType = None,
                 discriminator_S: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 Prompt_backbone: ConfigType = None,
                 Prompt_head: ConfigType = None,
                 Prompt_EMA: ConfigType = None,
                 SAM_EMA: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        ## WARNING by LYU: 2024/07/27 
        ## DO NOT BUILD BACKBONE FROM CONFIG
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        
        ## WARNING by LYU: 2024/07/27 
        ## NOT SUPPORT AUXILIARY HEAD
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        ## modified by LYU: 2024/07/11
        ## Here 'backbone' indicated the SAM model
        self.backbone = self.init_SAM_model(SAM_config, SAM_arch)

        ## added by LYU: 2025/04/08
        #######################
        self.lora_backbone = LoRA_Sam(self.backbone, 4)
        #######################

        ## modified by LYU: 2024/08/01
        if discriminator_P is not None:
            self.discriminator_P = MODELS.build(discriminator_P)
        if discriminator_S is not None:
            self.discriminator_S = MODELS.build(discriminator_S)
        ## added by LYU: 2024/07/29
        if Prompt_backbone is not None:
            self.Prompt_backbone = MODELS.build(Prompt_backbone)
        if Prompt_head is not None:
            self.Prompt_head = MODELS.build(Prompt_head)
        ## added by LYU: 2024/08/06
        if SAM_EMA is not None:
            self.SAM_EMA = SAM_EMA
            self.SAM_EMA_hyper = self._init_EMA(SAM_EMA)
            self.SAM_EMA_neck = MODELS.build(SAM_EMA['neck_EMA'])
            self.SAM_EMA_head = MODELS.build(SAM_EMA['decode_head_EMA']) 
        else:
            self.SAM_EMA = None
        ## added by LYU: 2024/08/05
        if Prompt_EMA is not None:
            self.Prompt_EMA = Prompt_EMA
            self.Prompt_EMA_hyper = self._init_EMA(Prompt_EMA)
            self.Prompt_EMA_backbone = MODELS.build(Prompt_EMA['backbone_EMA'])
            self.Prompt_EMA_head = MODELS.build(Prompt_EMA['decode_head_EMA']) 
        else:
            self.Prompt_EMA = None
    ##############################
    ## added by LYU: 2024/08/05
    def _init_EMA(self, cfg):
        output_param = dict()
        EMA_alpha = cfg['decay']
        EMA_training_ratio = cfg['training_ratio']
        EMA_pseu_cls_weight = cfg['pseudo_class_weight']
        EMA_pseu_thre = cfg['pseudo_threshold']
        EMA_rare_pseu_thre = cfg['pseudo_rare_threshold']
        output_param['EMA_alpha'] = EMA_alpha
        output_param['EMA_training_ratio'] = EMA_training_ratio
        output_param['EMA_pseu_cls_weight'] = EMA_pseu_cls_weight
        output_param['EMA_pseu_thre'] = EMA_pseu_thre
        output_param['EMA_rare_pseu_thre'] = EMA_rare_pseu_thre
        return output_param
    def _update_Prompt_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.Prompt_EMA_hyper['EMA_alpha'])
        ## 1. update target_backbone
        for ema_b, target_b in zip(self.Prompt_EMA_backbone.parameters(), self.Prompt_backbone.parameters()):
            ## For scalar params
            if not target_b.data.shape:
                ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
            ## For tensor params
            else:
                ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]
        ## 2. update target_decoder
        for ema_d, target_d in zip(self.Prompt_EMA_head.parameters(), self.Prompt_head.parameters()):
            ## For scalar params
            if not target_d.data.shape:
                ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
            ## For tensor params
            else:
                ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]
    def _update_SAM_EMA(self, iter):
        alpha_t = min(1 - 1 / (iter + 1), self.SAM_EMA_hyper['EMA_alpha'])
        ## 1. update SAM neck
        for ema_b, target_b in zip(self.SAM_EMA_neck.parameters(), self.neck.parameters()):
            ## For scalar params
            if not target_b.data.shape:
                ema_b.data = alpha_t * ema_b.data + (1 - alpha_t) * target_b.data
            ## For tensor params
            else:
                ema_b.data[:] = alpha_t * ema_b.data[:] + (1 - alpha_t) * target_b.data[:]
        ## 2. update target_decoder
        for ema_d, target_d in zip(self.SAM_EMA_head.parameters(), self.decode_head.parameters()):
            ## For scalar params
            if not target_d.data.shape:
                ema_d.data = alpha_t * ema_d.data + (1 - alpha_t) * target_d.data
            ## For tensor params
            else:
                ema_d.data[:] = alpha_t * ema_d.data[:] + (1 - alpha_t) * target_d.data[:]
    def pseudo_label_generation_EMA(self, pred, EMA_hyper, dev=None):
        ##############################
        #### 1. vanilla pseudo label generation
        pred_softmax = torch.softmax(pred, dim=1)
        pseudo_prob, pseudo_label = torch.max(pred_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(EMA_hyper['EMA_pseu_thre']).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight_ratio = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight_ratio * torch.ones(pseudo_prob.shape, device=dev)
        ##############################
        ##############################
        #### 2. class balancing strategy
        #### 2.1 change pseudo_weight and further set a threshold for rare class. E.g. For threshold over 0.8: 10x for car and clutter; 5x for 'low_vegetation' and 'tree'
        if EMA_hyper['EMA_pseu_cls_weight'] is not None and EMA_hyper['EMA_rare_pseu_thre'] is not None:
            ps_large_p_rare = pseudo_prob.ge(EMA_hyper['EMA_rare_pseu_thre']).long() == 1
            pseudo_weight = pseudo_weight * ps_large_p_rare
            pseudo_class_weight = copy.deepcopy(pseudo_label.float())
            for i in range(len(EMA_hyper['EMA_pseu_cls_weight'])):
                pseudo_class_weight[pseudo_class_weight == i] = EMA_hyper['EMA_pseu_cls_weight'][i]
            pseudo_weight = pseudo_class_weight * pseudo_weight
            pseudo_weight[pseudo_weight == 0] = pseudo_weight_ratio * 0.5
        ##############################
        pseudo_label = pseudo_label[:, None, :, :]
        return pseudo_label, pseudo_weight
    ##############################

    ## added by LYU: 2024/07/11
    def init_SAM_model(self, cfg: str, arch: str):
        model = MODELS.build(cfg)
        load_checkpoint(model, model_zoo.get(arch), strict=True)
        return model
    
    ## added by LYU: 2024/07/11
    def extract_feat_SAM(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        image_embeddings = self.backbone.module.image_encoder(inputs)
        return image_embeddings
    
    ## added by LYU: 2025/04/09
    def extract_feat_SAM_Lora(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        image_embeddings = self.lora_backbone.module.sam.image_encoder(inputs)
        return image_embeddings

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        ## added by LYU: 2024/08/14
        if not hasattr(self.backbone, 'module'):
            '''
            ## naive version
            x = self.backbone.image_encoder(inputs)
            if self.with_neck:
                f = self.neck(x)
            seg_logits = self.decode_head.predict(f, batch_img_metas,
                                              self.test_cfg)
            x = self.Prompt_backbone(inputs)
            Prompt_seg_logits = self.Prompt_head.predict(x, batch_img_metas,
                                              self.test_cfg)
            seg_logits = seg_logits + Prompt_seg_logits
            '''
            x = self.Prompt_backbone(inputs)
            Prompt_seg_logits = self.Prompt_head.predict(x, batch_img_metas,
                                              self.test_cfg)
            Prompt_seg_logits_forSAM = self.Prompt_head(x)
            Prompt_seg_mask = Prompt_seg_logits_forSAM.argmax(dim=1) * 1.0
            Prompt_seg_mask = Prompt_seg_mask[:, None, :, :]
            ## sparse embeddings for the points and boxes, dense embeddings for the masks,
            sparse_embeddings, dense_embeddings = self.backbone.prompt_encoder(
                points=None,
                boxes=None,
                masks=Prompt_seg_mask,
            )
            SAM_enc = self.backbone.image_encoder(inputs)
            low_res_masks, iou_predictions = self.backbone.mask_decoder(
                image_embeddings=SAM_enc[-1],
                image_pe=self.backbone.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            masks_t = F.interpolate(low_res_masks, (inputs.shape[2:]), mode='bilinear', align_corners=False)
            #self.Vis_Mask_decoder_out(masks_t)
            
            if self.with_neck:
                f = self.neck(SAM_enc)
            FD_seg_logits = self.decode_head(f, low_res_masks)
            FD_seg_logits = self.decode_head.predict_by_feat(FD_seg_logits, batch_img_metas)
            seg_logits = torch.softmax(FD_seg_logits, dim=1) + torch.softmax(Prompt_seg_logits, dim=1)
            #seg_logits = torch.softmax(Prompt_seg_logits+FD_seg_logits, dim=1) + torch.softmax(Prompt_seg_logits, dim=1)
            
            return seg_logits
        ## modified by LYU: 2024/07/11
        ## Noted by LYU: 2024/08/01
        ## 1. SAM segmentor inference
        x = self.extract_feat_SAM_Lora(inputs)
        if self.with_neck:
            f = self.neck(x)
        seg_logits = self.decode_head.module.predict(f, batch_img_metas,
                                              self.test_cfg)
        ## 2. Prompt segmentor inference
        x = self.Prompt_backbone.module(inputs)
        Prompt_seg_logits = self.Prompt_head.module.predict(x, batch_img_metas,
                                              self.test_cfg)
        seg_logits = seg_logits + Prompt_seg_logits
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        #x = self.extract_feat(inputs)
        ## modified by LYU: 2024/07/11
        x = self.extract_feat_SAM_Lora(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        ## modified by LYU: 2025.4.2
        #x = self.extract_feat(inputs)
        #return self.decode_head.forward(x)
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    ## added by LYU: 2024/07/25
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)

        ## added by LYU: 2024/08/05
        if curr_iter > 0:
            if self.Prompt_EMA is not None:
                self._update_Prompt_EMA(curr_iter)
            if self.SAM_EMA is not None:
                self._update_SAM_EMA(curr_iter)

        log_vars = dict()

        ## Step0: Frozen all trainable parameters & set all params as 'zero_grad'
        optim_wrapper.zero_grad()
        #self.set_requires_grad(self.backbone, False)
        self.set_requires_grad(self.neck, False)
        self.set_requires_grad(self.decode_head, False)
        self.set_requires_grad(self.discriminator_P, False)
        self.set_requires_grad(self.discriminator_S, False)
        self.set_requires_grad(self.Prompt_backbone, False)
        self.set_requires_grad(self.Prompt_head, False)

        ## Step1: Optimize segmentor
        ## Tips: Frozen SAM encoder & discriminator 
        #self.set_requires_grad(self.backbone, True)
        self.set_requires_grad(self.neck, True)
        self.set_requires_grad(self.decode_head, True)
        self.set_requires_grad(self.Prompt_backbone, True)
        self.set_requires_grad(self.Prompt_head, True)

        ## 1.0 forward SAM segmentor
        #SAM_backbone_out = self.SAM_backbone_forward(data['inputs'], data['B_inputs'])
        SAM_backbone_out = self.SAM_backbone_forward_Lora(data['inputs'], data['B_inputs'])
        
        ## 1.1 Optimize Prompt segmentor
        Prompt_segmentor_out = self.Prompt_segmentor_forward(data['inputs'], data['B_inputs'])
        loss_Prompt_seg = self.Prompt_head.module.loss_by_feat(Prompt_segmentor_out['Ppred_s'], data['data_samples'])
        parsed_losses_Prompt_seg, log_vars_Prompt_seg = self.parse_losses(loss_Prompt_seg)  # type: ignore
        log_vars_Prompt_seg['loss_ce_Prompt_seg'] = log_vars_Prompt_seg.pop('loss_ce')
        log_vars_Prompt_seg['acc_Prompt_seg'] = log_vars_Prompt_seg.pop('acc_seg')
        log_vars.update(log_vars_Prompt_seg)
        #####################################
        ## CODE for Prompt_EMA
        if self.Prompt_EMA_hyper is not None:
            EMA_fPb_t = self.Prompt_EMA_backbone.module(data['B_inputs'])
            EMA_Ppred_t = self.Prompt_EMA_head.module(EMA_fPb_t)
            EMA_Ppred_t = resize(
                input=EMA_Ppred_t,
                size=data['B_inputs'].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            EMA_Ppred_t_detach = EMA_Ppred_t.detach()
            Ppseudo_label, Ppseudo_weight = self.pseudo_label_generation_EMA(EMA_Ppred_t_detach, self.Prompt_EMA_hyper, data['B_inputs'].device)
            loss_Prompt_seg_EMA = self.Prompt_head.module.loss_by_feat(Prompt_segmentor_out['Ppred_t'], data['data_samples'], Ppseudo_label, Ppseudo_weight)
            parsed_losses_PromptEMA_seg, log_vars_PromptEMA_seg = self.parse_losses(loss_Prompt_seg_EMA)  # type: ignore
            log_vars_PromptEMA_seg['loss_ce_PromptEMA_seg'] = log_vars_PromptEMA_seg.pop('loss_ce')
            log_vars_PromptEMA_seg['acc_PromptEMA_seg'] = log_vars_PromptEMA_seg.pop('acc_seg')
            log_vars.update(log_vars_PromptEMA_seg)
            loss_Prompt = parsed_losses_Prompt_seg + self.Prompt_EMA_hyper['EMA_training_ratio'] * parsed_losses_PromptEMA_seg
        else:
            loss_Prompt = parsed_losses_Prompt_seg
        #####################################
        ## added by LYU: 2024/08/02
        ## feature-level adv on Prompt Segmentor
        dis_outputs = self.fea_adv_forward(Prompt_segmentor_out['Pfb_s'], Prompt_segmentor_out['Pfb_t'], self.discriminator_P, data)
        loss_adv_dis, log_vars_adv_dis = self._get_gan_loss(dis_outputs['f_t_dis'], 'Pfb_t_dis', 1, self.discriminator_P)
        loss_Prompt_stage1 = loss_Prompt + loss_adv_dis
        loss_Prompt_stage1.backward()
        log_vars_adv_dis['loss_gan_Pfb_t_dis'] = log_vars_adv_dis.pop('loss_gan_Pfb_t_dis')
        log_vars.update(log_vars_adv_dis)
        optim_wrapper['Prompt_backbone'].step()
        optim_wrapper['Prompt_head'].step()
        ## Optimize discriminator_P
        self.set_requires_grad(self.discriminator_P, True)
        if isinstance(Prompt_segmentor_out['Pfb_t'], tuple) or isinstance(Prompt_segmentor_out['Pfb_t'], list):
            Prompt_segmentor_out['Pfb_t'] = Prompt_segmentor_out['Pfb_t'][-1]
            Prompt_segmentor_out['Pfb_s'] = Prompt_segmentor_out['Pfb_s'][-1]
        else:
            Prompt_segmentor_out['Pfb_t'] = Prompt_segmentor_out['Pfb_t']
            Prompt_segmentor_out['Pfb_s'] = Prompt_segmentor_out['Pfb_s']
        Pfb_t_dis = Prompt_segmentor_out['Pfb_t'].detach()
        Pfb_s_dis = Prompt_segmentor_out['Pfb_s'].detach()
        dis_outputs = self.fea_adv_forward(Pfb_s_dis, Pfb_t_dis, self.discriminator_P, data)
        loss_adv_ds, log_vars_adv_ds = self._get_gan_loss(dis_outputs['f_s_dis'], 'f_s_dis_d', 1, self.discriminator_P)
        loss_adv_ds.backward()
        log_vars_adv_ds['loss_gan_Pfb_s_dis_d'] = log_vars_adv_ds.pop('loss_gan_f_s_dis_d')
        log_vars.update(log_vars_adv_ds)
        loss_adv_dt, log_vars_adv_dt = self._get_gan_loss(dis_outputs['f_t_dis'], 'f_t_dis_d', 0, self.discriminator_P)
        loss_adv_dt.backward()
        log_vars_adv_dt['loss_gan_Pfb_t_dis_d'] = log_vars_adv_dt.pop('loss_gan_f_t_dis_d')
        log_vars.update(log_vars_adv_dt)
        optim_wrapper['discriminator_P'].step()
        self.set_requires_grad(self.Prompt_backbone, False)
        self.set_requires_grad(self.Prompt_head, False)
        self.set_requires_grad(self.discriminator_P, False)

        if curr_iter > 500:
            ## 1.2 Forward Prompt encoder & Mask Decoder
            PE_output_emb = self.Prompt_encoder_forward(Prompt_segmentor_out, data['data_samples'][0].ori_shape)
            MD_output_mask = self.Mask_decoder_forward(SAM_backbone_out, PE_output_emb, data['data_samples'][0].ori_shape)
            SAM_segmentor_out = self.SAM_wPrompt_forward(SAM_backbone_out, MD_output_mask) 
            loss_SAM_seg = self.decode_head.module.loss_by_feat(SAM_segmentor_out['pred_s'], data['data_samples'])
            parsed_losses_SAM_seg, log_vars_SAM_seg = self.parse_losses(loss_SAM_seg)  # type: ignore
            log_vars.update(log_vars_SAM_seg)
            #parsed_losses_SAM_seg.backward()
            #####################################
            ## CODE for SAM_EMA
            if self.SAM_EMA_hyper is not None:
                EMA_fb_t = self.extract_feat_SAM_Lora(data['B_inputs'])
                EMA_fn_t = self.SAM_EMA_neck(EMA_fb_t)
                EMA_pred_t = self.SAM_EMA_head.module(EMA_fn_t)
                EMA_pred_t = resize(
                    input=EMA_pred_t,
                    size=data['B_inputs'].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                EMA_pred_t_detach = EMA_pred_t.detach()
                Spseudo_label, Spseudo_weight = self.pseudo_label_generation_EMA(EMA_pred_t_detach, self.SAM_EMA_hyper, data['B_inputs'].device)
                loss_SAM_seg_EMA = self.decode_head.module.loss_by_feat(SAM_segmentor_out['pred_t'], data['data_samples'], Spseudo_label, Spseudo_weight)
                parsed_losses_SAMEMA_seg, log_vars_SAMEMA_seg = self.parse_losses(loss_SAM_seg_EMA)  # type: ignore
                log_vars_SAMEMA_seg['loss_ce_SAMEMA_seg'] = log_vars_SAMEMA_seg.pop('loss_ce')
                log_vars_SAMEMA_seg['acc_SAMEMA_seg'] = log_vars_SAMEMA_seg.pop('acc_seg')
                log_vars.update(log_vars_SAMEMA_seg)
                loss_SAM = parsed_losses_SAM_seg + self.SAM_EMA_hyper['EMA_training_ratio'] * parsed_losses_SAMEMA_seg
            else:
                loss_SAM = parsed_losses_SAM_seg
            loss_SAM.backward()
            optim_wrapper['neck'].step()
            optim_wrapper['decode_head'].step()
            optim_wrapper['backbone'].step()
            #####################################

        return log_vars
    
    ## added by LYU: 2024/07/26
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad= requires_grad
    
    ## added by LYU: 2024/07/27
    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H*W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out
    
    ## added by LYU: 2024/07/31
    def SAM_backbone_forward(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat_SAM(img)
        fb_t = self.extract_feat_SAM(B_img)
        outputs['fb_s'] = fb_s
        outputs['fb_t'] = fb_t
        return outputs
    
    ## added by LYU: 2025/04/08
    def SAM_backbone_forward_Lora(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat_SAM_Lora(img)
        fb_t = self.extract_feat_SAM_Lora(B_img)
        outputs['fb_s'] = fb_s
        outputs['fb_t'] = fb_t
        return outputs
    
    ## added by LYU: 2024/07/31
    def SAM_wPrompt_forward(self, SAM_backbone_f, Prompt_m):
        outputs = dict()
        if self.with_neck:
            fn_s = self.neck(SAM_backbone_f['fb_s'])
            fn_t = self.neck(SAM_backbone_f['fb_t'])
        pred_s = self.decode_head(fn_s, Prompt_m['low_res_masks_s'])
        pred_t = self.decode_head(fn_t, Prompt_m['low_res_masks_t'])
        outputs['pred_s'] = pred_s
        outputs['pred_t'] = pred_t
        return outputs

    ## added by LYU: 2024/07/27
    def SAM_segmentor_forward(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat_SAM(img)
        fb_t = self.extract_feat_SAM(B_img)
        if self.with_neck:
            fn_s = self.neck(fb_s)
            fn_t = self.neck(fb_t)
        pred_s = self.decode_head(fn_s)
        pred_t = self.decode_head(fn_t)
        outputs['fb_s'] = fb_s
        outputs['fb_t'] = fb_t
        outputs['pred_s'] = pred_s
        outputs['pred_t'] = pred_t
        return outputs
    
    ## added by LYU: 2025/04/09
    def SAM_segmentor_forward_Lora(self, img, B_img):
        outputs = dict()
        fb_s = self.extract_feat_SAM_Lora(img)
        fb_t = self.extract_feat_SAM_Lora(B_img)
        if self.with_neck:
            fn_s = self.neck(fb_s)
            fn_t = self.neck(fb_t)
        pred_s = self.decode_head(fn_s)
        pred_t = self.decode_head(fn_t)
        outputs['fb_s'] = fb_s
        outputs['fb_t'] = fb_t
        outputs['pred_s'] = pred_s
        outputs['pred_t'] = pred_t
        return outputs
    
    ## added by LYU: 2024/07/29
    def Prompt_segmentor_forward(self, img, B_img):
        outputs = dict()
        fPb_s = self.Prompt_backbone.module(img)
        fPb_t = self.Prompt_backbone.module(B_img)
        Ppred_s = self.Prompt_head.module(fPb_s)
        Ppred_t = self.Prompt_head.module(fPb_t)
        outputs['Pfb_s'] = fPb_s
        outputs['Pfb_t'] = fPb_t
        outputs['Ppred_s'] = Ppred_s
        outputs['Ppred_t'] = Ppred_t
        return outputs
    
    ## added by LYU: 2024/07/29
    def Prompt_encoder_forward(self, Prompt_segmentor_oup, size):
        outputs = dict()
        Prompt_seg_logits_s = Prompt_segmentor_oup['Ppred_s']
        Prompt_seg_logits_t = Prompt_segmentor_oup['Ppred_t']
        Prompt_seg_mask_s = Prompt_seg_logits_s.argmax(dim=1) * 1.0
        Prompt_seg_mask_t = Prompt_seg_logits_t.argmax(dim=1) * 1.0
        Prompt_seg_mask_s = Prompt_seg_mask_s[:, None, :, :]
        Prompt_seg_mask_t = Prompt_seg_mask_t[:, None, :, :]
        ## sparse embeddings for the points and boxes, dense embeddings for the masks,
        sparse_embeddings_s, dense_embeddings_s = self.backbone.module.prompt_encoder(
                points=None,
                boxes=None,
                masks=Prompt_seg_mask_s,
            )
        sparse_embeddings_t, dense_embeddings_t = self.backbone.module.prompt_encoder(
                points=None,
                boxes=None,
                masks=Prompt_seg_mask_t,
            )
        outputs['se_s'] = sparse_embeddings_s 
        outputs['se_t'] = sparse_embeddings_t
        outputs['de_s'] = dense_embeddings_s 
        outputs['de_t'] = dense_embeddings_t
        return outputs
    
    ## added by LYU: 2024/07/29
    def Mask_decoder_forward(self, SAM_backbone_oup, Prompt_encoder_oup, size):
        outputs = dict()
        if isinstance(SAM_backbone_oup['fb_t'], tuple) or isinstance(SAM_backbone_oup['fb_t'], list):
            seg_fb_t = SAM_backbone_oup['fb_t'][-1]
            seg_fb_s = SAM_backbone_oup['fb_s'][-1]
        else:
            seg_fb_t = SAM_backbone_oup['fb_t']
            seg_fb_s = SAM_backbone_oup['fb_s']
        low_res_masks_s, iou_predictions_s = self.backbone.module.mask_decoder(
            image_embeddings=seg_fb_s,
            image_pe=self.backbone.module.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=Prompt_encoder_oup['se_s'],
            dense_prompt_embeddings=Prompt_encoder_oup['de_s'],
            multimask_output=True,
        )
        low_res_masks_t, iou_predictions_t = self.backbone.module.mask_decoder(
            image_embeddings=seg_fb_t,
            image_pe=self.backbone.module.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=Prompt_encoder_oup['se_t'],
            dense_prompt_embeddings=Prompt_encoder_oup['de_t'],
            multimask_output=True,
        )
        masks_s = F.interpolate(
            low_res_masks_s, size, mode='bilinear', align_corners=False)
        masks_t = F.interpolate(
            low_res_masks_t, size, mode='bilinear', align_corners=False)
        outputs['masks_s'] = masks_s 
        outputs['masks_t'] = masks_t
        outputs['low_res_masks_s'] = low_res_masks_s 
        outputs['low_res_masks_t'] = low_res_masks_t
        return outputs

    ## added by LYU: 2024/07/29
    @staticmethod
    def Vis_Mask_decoder_out(mask):
        import matplotlib.pyplot as plt
        #vis_mask = torch.max(mask, dim=1)[0]
        vis_mask = torch.mean(mask, dim=1)
        print(vis_mask.shape)
        #vis_mask = (vis_mask - vis_mask.mean())/(vis_mask.std())
        #vis_mask = self.sw_softmax(mask)
        vis_mask = vis_mask[0, :, :].detach().cpu().numpy()
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.imshow(mask[0, 0, :, :].detach().cpu().numpy())
        plt.subplot(222)
        plt.imshow(mask[0, 1, :, :].detach().cpu().numpy())
        plt.subplot(223)
        plt.imshow(mask[0, 2, :, :].detach().cpu().numpy())
        plt.subplot(224)
        plt.imshow(vis_mask)
        plt.savefig("test.png")

    ## added by LYU: 2024/07/27
    def discriminator_fea_forward(self, seg_pred, discriminator):
        outputs = dict()
        outputs['f_s_dis'] = discriminator(seg_pred['f_s'])
        outputs['f_t_dis'] = discriminator(seg_pred['f_t'])
        return outputs

    ## added by LYU: 2024/07/27
    def discriminator_logits_forward(self, seg_pred, discriminator):
        outputs = dict()
        outputs['pred_s_dis'] = discriminator(seg_pred['pred_s'])
        outputs['pred_t_dis'] = discriminator(seg_pred['pred_t'])
        return outputs
    
    ## added by LYU: 2024/07/27
    def _get_gan_loss(self, pred, domain, target_is_real, discriminator):
        losses = dict()
        losses[f'loss_gan_{domain}'] = discriminator.module.gan_loss(pred, target_is_real)
        loss_g, log_vars_g = self.parse_losses(losses)
        return loss_g, log_vars_g
    
    ## added by LYU: 2024/08/02
    def logits_adv_forward(self, seg_outputs_s, seg_outputs_t, discriminator, data):
        seg_output_adv = dict()
        if isinstance(seg_outputs_t, tuple):
            seg_pred_t = self.sw_softmax(seg_outputs_t[-1])
            seg_pred_s = self.sw_softmax(seg_outputs_s[-1])
        else:
            seg_pred_t = self.sw_softmax(seg_outputs_t)
            seg_pred_s = self.sw_softmax(seg_outputs_s)
        seg_output_adv['pred_t'] = seg_pred_t
        seg_output_adv['pred_s'] = seg_pred_s
        dis_outputs = self.discriminator_logits_forward(seg_output_adv, discriminator)
        dis_outputs['pred_t_dis'] = resize(
            input=dis_outputs['pred_t_dis'],
            size=data['B_inputs'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dis_outputs['pred_s_dis'] = resize(
            input=dis_outputs['pred_s_dis'],
            size=data['inputs'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return dis_outputs
    
    ## added by LYU: 2024/08/02
    def fea_adv_forward(self, seg_outputs_s, seg_outputs_t, discriminator, data):
        seg_output_adv = dict()
        if isinstance(seg_outputs_t, tuple) or isinstance(seg_outputs_t, list):
            seg_f_t = self.sw_softmax(seg_outputs_t[-1])
            seg_f_s = self.sw_softmax(seg_outputs_s[-1])
        else:
            seg_f_t = self.sw_softmax(seg_outputs_t)
            seg_f_s = self.sw_softmax(seg_outputs_s)
        seg_output_adv['f_t'] = seg_f_t
        seg_output_adv['f_s'] = seg_f_s
        dis_outputs = self.discriminator_fea_forward(seg_output_adv, discriminator)
        dis_outputs['f_t_dis'] = resize(
            input=dis_outputs['f_t_dis'],
            size=data['B_inputs'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dis_outputs['f_s_dis'] = resize(
            input=dis_outputs['f_s_dis'],
            size=data['inputs'].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return dis_outputs


## added by LYU: 2025/04/08
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model, r, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.layers)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.layers):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    '''
    def forward(self, batched_input, multimask_output, image_size):
        sam_output=self.sam(batched_input, multimask_output, image_size)
        img_feature=sam_output['encoder_feature']
        return sam_output
    '''