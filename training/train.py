"""
train.py — Mask R-CNN fine-tuning on the leaf instance segmentation dataset.

Run prepare_dataset.py first to merge per-set LabelMe-derived COCO files
into data/train.json and data/val.json. Image paths in those JSONs are
relative to OC1_DATASET_ROOT (env var) so they resolve via image_root.

Run:
    python train.py                    # train
    python train.py --resume           # resume from last checkpoint
    python train.py --eval-only        # evaluation only
    python train.py --num-gpus 2       # multi-GPU
"""

import os
import argparse
import logging

from detectron2.utils.logger import setup_logger
setup_logger()

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T
from detectron2.config import get_cfg as _d2_get_cfg

from maskrcnn_custom_config import get_cfg

logger = logging.getLogger("detectron2")


# ---------------------------------------------------------------------------
# Dataset registration
# ---------------------------------------------------------------------------

DATASET_ROOT = os.environ.get("OC1_DATASET_ROOT", "./dataset")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def register_datasets():
    """Register merged COCO datasets produced by prepare_dataset.py."""
    meta = {"thing_classes": ["plant", "leaf"]}
    register_coco_instances(
        name="oc1_train",
        metadata=meta,
        json_file=os.path.join(DATA_DIR, "train.json"),
        image_root=DATASET_ROOT,
    )
    register_coco_instances(
        name="oc1_val",
        metadata=meta,
        json_file=os.path.join(DATA_DIR, "val.json"),
        image_root=DATASET_ROOT,
    )


# ---------------------------------------------------------------------------
# Custom augmentation mapper (optional)
# ---------------------------------------------------------------------------

def build_train_aug(cfg):
    """
    Build an augmentation list for training.
    Extend this with more augmentations as needed.
    """
    augs = [
        T.ResizeShortestEdge(
            short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
            max_size=cfg.INPUT.MAX_SIZE_TRAIN,
            sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        ),
        T.RandomFlip(horizontal=True, vertical=False),
        # ----  Optional extra augmentations  ----
        # T.RandomBrightness(0.8, 1.2),
        # T.RandomContrast(0.8, 1.2),
        # T.RandomSaturation(0.8, 1.2),
        # T.RandomRotation(angle=[-10, 10]),
        # T.RandomCrop("relative_range", (0.85, 0.85)),
    ]
    return augs


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

class CustomTrainer(DefaultTrainer):
    """
    Extends DefaultTrainer with:
      - Custom augmentation pipeline
      - COCO evaluator on validation set
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=build_train_aug(cfg),
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        evaluators = [
            COCOEvaluator(
                dataset_name,
                tasks=("bbox", "segm"),   # evaluate boxes + masks
                distributed=True,
                output_dir=output_folder,
            )
        ]
        return DatasetEvaluators(evaluators)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup(args):
    cfg = get_cfg()

    # Allow command-line overrides (e.g. --opts SOLVER.MAX_ITER 5000)
    if args.opts:
        cfg = _d2_get_cfg()
        cfg.merge_from_file(
            __import__("detectron2").model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.merge_from_list(args.opts)
    else:
        cfg = get_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)
    return cfg


def main(args):
    register_datasets()
    cfg = setup(args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        return res

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN fine-tuning")
    parser.add_argument("--resume",    action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--num-gpus",  type=int, default=1,  help="Number of GPUs")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank",  type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Override config options using key=value pairs")
    args = parser.parse_args()

    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
