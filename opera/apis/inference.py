# Copyright (c) Hikvision Research Institute. All rights reserved.
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose

from opera.datasets import replace_ImageToTensor
from opera.models import build_model


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    from mmcv.parallel import DataContainer as DC

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # Make a copy of the config to modify the pipeline
    cfg = cfg.copy()

    # Check if this is a multi-frame model by looking at the pipeline
    is_multi_frame = False
    if len(cfg.data.test.pipeline) > 0:
        first_type = cfg.data.test.pipeline[0].get('type', '')
        is_multi_frame = 'LoadMul' in first_type

    if is_multi_frame:
        # For multi-frame models, create a custom pipeline that handles
        # img_prev, img_now, img_next properly
        # Extract transforms from MultiScaleFlipAug if present
        img_scale = (1333, 800)
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )

        # Try to get img_scale and normalization config from the pipeline
        for pipeline_step in cfg.data.test.pipeline:
            if pipeline_step.get('type', '') == 'mmdet.MultiScaleFlipAug':
                img_scale = pipeline_step.get('img_scale', img_scale)
                for transform in pipeline_step.get('transforms', []):
                    if 'Normalize' in transform.get('type', ''):
                        img_norm_cfg = {
                            'mean': transform.get('mean', img_norm_cfg['mean']),
                            'std': transform.get('std', img_norm_cfg['std']),
                            'to_rgb': transform.get('to_rgb', img_norm_cfg['to_rgb'])
                        }
                        break
                break

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                img_array = img.astype(np.float32)
                filename = 'numpy_array'
            else:
                # Load image from file
                img_array = mmcv.imread(img).astype(np.float32)
                filename = img

            # Process each frame: resize, normalize, pad
            def process_frame(frame, img_scale, img_norm_cfg):
                """Process a single frame with resize, normalize, pad."""
                h, w = frame.shape[:2]
                max_long_edge = max(img_scale)
                max_short_edge = min(img_scale)
                scale_factor = min(max_long_edge / max(h, w),
                                   max_short_edge / min(h, w))
                new_w = int(w * scale_factor + 0.5)
                new_h = int(h * scale_factor + 0.5)

                # Resize
                frame = mmcv.imresize(frame, (new_w, new_h))

                # Normalize
                mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
                std = np.array(img_norm_cfg['std'], dtype=np.float32)
                if img_norm_cfg['to_rgb']:
                    frame = frame[..., ::-1].copy()  # BGR to RGB
                frame = (frame - mean) / std

                return frame, scale_factor, (new_h, new_w)

            # Process all three frames
            img_prev, scale_factor, new_shape = process_frame(
                img_array.copy(), img_scale, img_norm_cfg)
            img_now, _, _ = process_frame(
                img_array.copy(), img_scale, img_norm_cfg)
            img_next, _, _ = process_frame(
                img_array.copy(), img_scale, img_norm_cfg)

            # Convert to tensor format: (H, W, C) -> (C, H, W)
            def to_tensor_format(frame):
                if len(frame.shape) < 3:
                    frame = np.expand_dims(frame, -1)
                return torch.from_numpy(
                    np.ascontiguousarray(frame.transpose(2, 0, 1))
                ).float()

            img_prev_t = to_tensor_format(img_prev)
            img_now_t = to_tensor_format(img_now)
            img_next_t = to_tensor_format(img_next)

            # Stack frames: shape (3, C, H, W)
            img_tensor = torch.stack([img_prev_t, img_now_t, img_next_t])

            # Build img_metas
            # scale_factor should be [scale_w, scale_h, scale_w, scale_h] format
            scale_factor_array = np.array(
                [scale_factor, scale_factor, scale_factor, scale_factor],
                dtype=np.float32)
            img_metas = {
                'filename_now': filename,
                'ori_filename_now': filename,
                'ori_shape': img_array.shape,
                'img_shape': new_shape + (3,),
                'pad_shape': new_shape + (3,),
                'scale_factor': scale_factor_array,
                'flip': False,
                'flip_direction': None,
                'img_norm_cfg': img_norm_cfg,
            }

            data = {
                'img_metas': DC([[img_metas]], cpu_only=True),
                'img': DC([[img_tensor]])
            }
            datas.append(data)
    else:
        # Standard single-image pipeline
        if isinstance(imgs[0], np.ndarray):
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                data = dict(img=img)
            else:
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

    if is_multi_frame:
        # For multi-frame, manually combine the data
        img_metas_list = []
        img_list = []
        for d in datas:
            img_metas_list.append(d['img_metas'].data[0][0])
            img_list.append(d['img'].data[0][0])

        data = {
            'img_metas': [img_metas_list],
            'img': [torch.stack(img_list)]
        }
    else:
        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)


def inference_video(model, frames, score_thr=0.5, max_track_queries=100):
    """Run inference on a sequence of frames with track query propagation.

    The model must be a PAVE detector with tracking state support.

    Args:
        model (nn.Module): The loaded PAVE detector.
        frames (list[str or np.ndarray]): List of frame images (filenames or arrays).
        score_thr (float): Score threshold for track query selection.
        max_track_queries (int): Maximum number of track queries to propagate.

    Returns:
        list[tuple]: List of (bbox_result, keypoint_result) for each frame,
            along with track_ids for each detection.
    """
    if hasattr(model, 'module'):
        model = model.module

    model.reset_tracking_state()
    results = []

    for frame in frames:
        result = inference_detector(model, frame)
        # Collect track IDs if available
        track_ids = getattr(model, 'track_ids', None)
        results.append((result, track_ids))

    return results
