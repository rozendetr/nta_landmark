# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import pandas as pd


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.to("cuda")
    # print(model)
    # load model
    # state_dict = torch.load(args.model_file)
    # print(state_dict)
    # model = torch.load(args.model_file)
    with open(args.model_file, "rb") as fp:
        state_dict = torch.load(fp)
        model.load_state_dict(state_dict)
    # model.load_state_dict(state_dict['state_dict'])
    # if 'state_dict' in state_dict.keys():
    #     state_dict = state_dict['state_dict']
    #     # print(state_dict)
    #     model.load_state_dict(state_dict)
    # else:
    #     model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    predictions  = function.inference(config, test_loader, model)
    # print("len(predictions)", len(predictions))
    # print(predictions[0])
    df_predictions = []
    for pred in predictions:
        row = dict()
        row['file_name'] = pred[0]
        for id_point in range(194):
            row[f'Point_M{id_point}_X'] = int(pred[1][id_point])
            row[f'Point_M{id_point}_Y'] = int(pred[2][id_point])
        df_predictions.append(row)
    df_predictions = pd.DataFrame(df_predictions)
    # print(predictions_meta[0])
    df_predictions.to_csv('pred_test.csv', index=False)



    # torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

