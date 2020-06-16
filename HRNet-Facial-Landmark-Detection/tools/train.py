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
import torch.optim as optim
from torch.nn import functional as fnn
# import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
import math


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args

# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target, reduction='mean'):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class AWing(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)

    def forward(self, y_pred , y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        return lossMat

class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)

    def forward(self, y_pred , y, M, reduction='mean'):
        M = M.float()
        Loss = self.Awing(y_pred,y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()



def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED



    # if isinstance(config.TRAIN.LR_STEP, list):
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, config.TRAIN.LR_STEP,
    #         # config.TRAIN.LR_FACTOR, last_epoch-1
    #         config.TRAIN.LR_FACTOR, 0
    #     )
    # else:
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, config.TRAIN.LR_STEP,
    #         # config.TRAIN.LR_FACTOR, last_epoch-1
    #     config.TRAIN.LR_FACTOR, 0
    #     )
    dataset_type = get_dataset(config)
    train_dataset = dataset_type(config,
                             is_train=True)

    # train_dataset[0]
    # return 0

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=True),
        # batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS)



    # val_loader = DataLoader(
    #     dataset=dataset_type(config,
    #                          is_train=True),
    #     # batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #     batch_size=config.TEST.BATCH_SIZE_PER_GPU,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     # pin_memory=config.PIN_MEMORY
    # )

    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.to("cuda")
    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    # criterion = fnn.mse_loss
    # criterion = WingLoss()
    # criterion = Loss_weighted()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')

        if os.path.isfile(model_state_file):
            with open(model_state_file, "rb") as fp:
                state_dict = torch.load(fp)
                model.load_state_dict(state_dict)
                last_epoch = 1
            # checkpoint = torch.load(model_state_file)
            # last_epoch = checkpoint['epoch']
            # best_nme = checkpoint['best_nme']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(last_epoch))
        else:
            print("=> no checkpoint found")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        # evaluate
        nme = 0
        # nme, predictions = function.validate(config, val_loader, model,
        #                                    criterion, epoch, writer_dict)

        is_best = True
        # is_best = nme < best_nme
        best_nme = min(nme, best_nme)


        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        torch.save(model.state_dict(), os.path.join(final_output_dir, 'mse_relu_lips_checkpoint_{}.pth'.format(epoch)))

        # utils.save_checkpoint(
        #     {"state_dict": model,
        #      "epoch": epoch + 1,
        #      "best_nme": best_nme,
        #      "optimizer": optimizer.state_dict(),
        #      }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()










