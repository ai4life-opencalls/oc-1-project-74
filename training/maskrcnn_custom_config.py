"""
Full Detectron2 Mask R-CNN Config for Custom Instance Segmentation
==================================================================
Fine-tuning from COCO-pretrained weights on a custom dataset.

Usage:
    python train.py --config-file maskrcnn_custom_config.py
    
Or load programmatically:
    from maskrcnn_custom_config import get_cfg
"""

from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo


def get_cfg():
    cfg = _get_cfg()

    # -------------------------------------------------------------------------
    # 1. BASE MODEL
    #    Load a Mask R-CNN baseline from the model zoo.
    #    Options:
    #      - mask_rcnn_R_50_FPN_3x.yaml   (faster, good baseline)
    #      - mask_rcnn_R_101_FPN_3x.yaml  (more capacity)
    #      - mask_rcnn_X_101_32x8d_FPN_3x.yaml (strongest, slower)
    # -------------------------------------------------------------------------
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # -------------------------------------------------------------------------
    # 2. DATASETS
    #    Register your dataset first via DatasetCatalog / MetadataCatalog.
    #    See train.py below for registration helpers.
    # -------------------------------------------------------------------------
    cfg.DATASETS.TRAIN = ("oc1_train",)
    cfg.DATASETS.TEST  = ("oc1_val",)
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST  = 1000

    # -------------------------------------------------------------------------
    # 3. DATALOADER
    # -------------------------------------------------------------------------
    cfg.DATALOADER.NUM_WORKERS         = 4      # CPU workers for data loading
    cfg.DATALOADER.SAMPLER_TRAIN       = "TrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD    = 0.0    # for RepeatFactorTrainingSampler
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # skip images with no annotations

    # -------------------------------------------------------------------------
    # 4. MODEL — BACKBONE
    # -------------------------------------------------------------------------
    cfg.MODEL.BACKBONE.NAME            = "build_resnet_fpn_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT       = 2  # freeze stem + res2; set 0 to train all

    # ResNet
    cfg.MODEL.RESNETS.DEPTH            = 50      # 50 or 101
    cfg.MODEL.RESNETS.OUT_FEATURES     = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NUM_GROUPS       = 1        # 1 = standard ResNet; 32 for ResNeXt
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP  = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1    = True
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.RES5_DILATION    = 1        # 1 = no dilation; 2 = dilated conv
    cfg.MODEL.RESNETS.NORM             = "FrozenBN"  # FrozenBN | BN | GN | SyncBN

    # FPN
    cfg.MODEL.FPN.IN_FEATURES          = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS         = 256
    cfg.MODEL.FPN.NORM                 = ""
    cfg.MODEL.FPN.FUSE_TYPE            = "sum"    # sum | avg

    # -------------------------------------------------------------------------
    # 5. MODEL — ANCHOR GENERATOR
    # -------------------------------------------------------------------------
    cfg.MODEL.ANCHOR_GENERATOR.NAME    = "DefaultAnchorGenerator"
    # One set of sizes per FPN level (P2–P6)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES   = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # per level
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET  = 0.0

    # -------------------------------------------------------------------------
    # 6. MODEL — RPN (Region Proposal Network)
    # -------------------------------------------------------------------------
    cfg.MODEL.RPN.HEAD_NAME            = "StandardRPNHead"
    cfg.MODEL.RPN.IN_FEATURES          = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.BOUNDARY_THRESH      = -1       # -1 = no clipping
    cfg.MODEL.RPN.IOU_THRESHOLDS       = [0.3, 0.7]   # bg / fg IoU thresholds
    cfg.MODEL.RPN.IOU_LABELS           = [0, -1, 1]    # bg, ignore, fg
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256      # proposals sampled per image
    cfg.MODEL.RPN.POSITIVE_FRACTION    = 0.5
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS     = (1.0, 1.0, 1.0, 1.0)
    cfg.MODEL.RPN.LOSS_WEIGHT          = 1.0

    # NMS thresholds
    cfg.MODEL.RPN.NMS_THRESH           = 0.7
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN   = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN  = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST    = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST   = 1000
    cfg.MODEL.RPN.MIN_SIZE             = 0

    # Regression loss
    cfg.MODEL.RPN.SMOOTH_L1_BETA       = 0.0     # 0.0 → pure L1; 1.0 → Smooth L1
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE   = "smooth_l1"  # smooth_l1 | giou

    # -------------------------------------------------------------------------
    # 7. MODEL — ROI HEADS
    # -------------------------------------------------------------------------
    cfg.MODEL.ROI_HEADS.NAME           = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES    = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 2        # plant, leaf
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]    # FG/BG split for proposals
    cfg.MODEL.ROI_HEADS.IOU_LABELS     = [0, 1]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # proposals per image for training
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION   = 0.25   # fraction of FG proposals
    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT  = True   # add GT boxes to proposals
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = 0.05   # detection confidence cutoff
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST     = 0.5
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100   # max detections per image

    # ROI Box Head
    cfg.MODEL.ROI_BOX_HEAD.NAME                  = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC                = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM                = 1024
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV              = 0
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM              = 256
    cfg.MODEL.ROI_BOX_HEAD.NORM                  = ""      # "" | GN | BN
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False   # class-specific regressor
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA        = 0.0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION     = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE           = "ROIAlignV2"  # ROIAlign | ROIAlignV2
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS      = (10.0, 10.0, 5.0, 5.0)
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE    = "smooth_l1"   # smooth_l1 | giou
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT  = 1.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS          = False   # Federated loss (open-vocab)
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE        = False

    # ROI Mask Head
    cfg.MODEL.ROI_MASK_HEAD.NAME                 = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV             = 4       # convs before upsampling
    cfg.MODEL.ROI_MASK_HEAD.CONV_DIM             = 256
    cfg.MODEL.ROI_MASK_HEAD.NORM                 = ""      # "" | GN
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION    = 14      # mask pool size
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE          = "ROIAlignV2"
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT          = 1.0
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK    = False   # class-specific masks

    # -------------------------------------------------------------------------
    # 8. PRETRAINED WEIGHTS
    #    Start from COCO-pretrained checkpoint; final layers will be re-init'd
    #    automatically when num_classes differs from 80.
    # -------------------------------------------------------------------------
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # -------------------------------------------------------------------------
    # 9. INPUT / AUGMENTATION
    # -------------------------------------------------------------------------
    cfg.INPUT.FORMAT                   = "BGR"    # BGR (default OpenCV) or RGB
    cfg.INPUT.MASK_FORMAT              = "polygon"  # polygon | bitmask

    # Resize: shorter edge randomly sampled from MIN_SIZE_TRAIN list
    cfg.INPUT.MIN_SIZE_TRAIN           = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN           = 1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING  = "choice"  # choice | range
    cfg.INPUT.MIN_SIZE_TEST            = 800
    cfg.INPUT.MAX_SIZE_TEST            = 1333

    # Pixel normalisation (ImageNet mean / std in BGR order)
    cfg.MODEL.PIXEL_MEAN               = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD                = [1.0, 1.0, 1.0]

    # Horizontal flip (only built-in aug in standard cfg; use custom mapper for more)
    cfg.INPUT.RANDOM_FLIP              = "horizontal"  # horizontal | vertical | none

    # Crop augmentation (optional, good for small objects)
    cfg.INPUT.CROP.ENABLED             = False
    cfg.INPUT.CROP.TYPE                = "relative_range"  # absolute | relative | relative_range
    cfg.INPUT.CROP.SIZE                = [0.9, 0.9]

    # -------------------------------------------------------------------------
    # 10. SOLVER (Optimizer + LR Schedule)
    # -------------------------------------------------------------------------
    # Rule of thumb for linear LR scaling: BASE_LR * (IMS_PER_BATCH / 16)
    # For fine-tuning, use ~10× smaller LR than pretraining.
    cfg.SOLVER.IMS_PER_BATCH           = 4        # total images/step across all GPUs
    cfg.SOLVER.BASE_LR                 = 0.00025  # 0.02 / 8 GPUs → scale down for 1 GPU
    cfg.SOLVER.MOMENTUM                = 0.9
    cfg.SOLVER.NESTEROV                = False
    cfg.SOLVER.WEIGHT_DECAY            = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM       = 0.0      # separate WD for norm layers
    cfg.SOLVER.BIAS_LR_FACTOR          = 1.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS       = 0.0
    cfg.SOLVER.OPTIMIZER               = "SGD"    # SGD | ADAM

    # LR schedule
    cfg.SOLVER.LR_SCHEDULER_NAME       = "WarmupMultiStepLR"
    cfg.SOLVER.MAX_ITER                = 10000    # ← tune to your dataset size
                                                   # rule: ~(dataset_size / IMS_PER_BATCH) * epochs
    cfg.SOLVER.STEPS                   = (7000, 9000)  # LR drop steps
    cfg.SOLVER.GAMMA                   = 0.1       # LR multiplied by GAMMA at each step
    cfg.SOLVER.WARMUP_FACTOR           = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS            = 1000
    cfg.SOLVER.WARMUP_METHOD           = "linear"  # linear | constant

    # Gradient clipping (optional but useful for small datasets)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED  = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"  # value | norm
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE  = 2.0

    # AMP (Automatic Mixed Precision — requires CUDA + PyTorch >= 1.6)
    cfg.SOLVER.AMP.ENABLED             = False    # set True to use fp16

    # Checkpoint saving
    cfg.SOLVER.CHECKPOINT_PERIOD       = 1000     # save every N iterations

    # -------------------------------------------------------------------------
    # 11. TEST / EVALUATION
    # -------------------------------------------------------------------------
    cfg.TEST.EVAL_PERIOD               = 1000     # run eval every N iters (0 = end only)
    cfg.TEST.EXPECTED_RESULTS          = []        # list of (task, metric, value, tolerance)
    cfg.TEST.PRECISE_BN.ENABLED        = False     # recompute BN stats before eval
    cfg.TEST.PRECISE_BN.NUM_ITER       = 200
    cfg.TEST.AUG.ENABLED               = False     # test-time augmentation
    cfg.TEST.AUG.MIN_SIZES             = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
    cfg.TEST.AUG.MAX_SIZE              = 4000
    cfg.TEST.AUG.FLIP                  = True
    cfg.TEST.DETECTIONS_PER_IMAGE      = 100

    # -------------------------------------------------------------------------
    # 12. OUTPUT
    # -------------------------------------------------------------------------
    cfg.OUTPUT_DIR                     = "./output/maskrcnn_custom"

    # -------------------------------------------------------------------------
    # 13. MISC
    # -------------------------------------------------------------------------
    cfg.SEED                           = 42        # -1 = random seed
    cfg.CUDNN_BENCHMARK                = False
    cfg.MODEL.DEVICE                   = "cuda"    # cuda | cpu

    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Quick helper: print the full config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_cfg()
    print(cfg.dump())   # YAML dump of the entire config
