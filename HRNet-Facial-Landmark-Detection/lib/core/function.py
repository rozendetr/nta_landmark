# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme

from PIL import Image, ImageFile
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)
        inp = inp.cuda()
        # compute the output
        output = model(inp)
        # print(target[0].shape)
        # Image.fromarray(np.uint8(target[0].numpy().sum(axis=0) * 255)).show()
        target = target.cuda(non_blocking=True)
        M = meta["M"].cuda()

        # loss = critertion(output, target, M, reduction="mean")
        loss = critertion(output, target)
        # print(loss)
        # NME
        score_map = output.data.cpu()

        # Image.fromarray(np.uint8(score_map[0].numpy().sum(axis=0) * 255)).show()
        # print(meta['pts'][0][:10])
        # print(meta['tpts'][0][:10])
        # print(decode_preds(target.data.cpu(), meta['center'], meta['scale'], [64, 64])[0][:10])

        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
        
        # print(output[0])
        # print(target.shape)
        # print(target[0][0])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.10f} '\
        .format(epoch, batch_time.avg, losses.avg)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            inp = inp.cuda()
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.10f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    # predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))
    # predictions_meta = []
    predictions = []
    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inp = inp.cuda()
            output = model(inp)
            print(output.shape)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            #
            # # NME
            # nme_temp = compute_nme(preds, meta)
            #
            # failure_008 = (nme_temp > 0.08).sum()
            # failure_010 = (nme_temp > 0.10).sum()
            # count_failure_008 += failure_008
            # count_failure_010 += failure_010
            #
            # nme_batch_sum += np.sum(nme_temp)
            # nme_count = nme_count + preds.size(0)


            pred_landmarks = preds.numpy().reshape((len(preds), 194, 2))
            file_name = meta['file_name']
            fs = meta["scale_coef"].numpy()  # B
            margins_x = meta["crop_margin_x"].numpy()  # B
            margins_y = meta["crop_margin_y"].numpy()  # B
            top_x = meta["top_x"].numpy()  # B
            top_y = meta["top_y"].numpy()  # B

            pred_landmarks[:, :, 0] += margins_x[:, None]
            pred_landmarks[:, :, 1] += margins_y[:, None]
            pred_landmarks /= fs[:, None, None]
            pred_landmarks[:, :, 0] += top_x[:, None]
            pred_landmarks[:, :, 1] += top_y[:, None]

            for i in range(data_loader.batch_size):
                row = [file_name[i], pred_landmarks[i, :, 0], pred_landmarks[i, :, 1]]
                predictions.append(row)

            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]
            # predictions_meta.append(meta)

                # predictions_meta.append(meta[n])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # nme = nme_batch_sum / nme_count
    # failure_008_rate = count_failure_008 / nme_count
    # failure_010_rate = count_failure_010 / nme_count
    #
    # msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
    #       '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
    #                             failure_008_rate, failure_010_rate)
    # logger.info(msg)

    return predictions



