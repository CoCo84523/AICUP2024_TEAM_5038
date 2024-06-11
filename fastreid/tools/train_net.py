#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fast_reid.fastreid.utils.checkpoint import Checkpointer

# from torch.utils.tensorboard import SummaryWriter


from torchinfo import summary
import torch


def setup(args):
    """
    Create configs and perform basic setups.
    """

    args.config_file = r'fast_reid/configs/AICUP/bagtricks_R50-ibn.yml'
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    if args.eval_only:
    # if 1:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = True
    
        model = DefaultTrainer.build_model(cfg)

        # print(model)
        # input_data = torch.ones(1, 3, 224, 224).cuda()

        # writer = SummaryWriter(f'aaa', filename_suffix='-model_graph.tb')
        # writer.add_graph(model, input_data)
        # writer.close()

        # summary(model, input_data.size())

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch( 
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
