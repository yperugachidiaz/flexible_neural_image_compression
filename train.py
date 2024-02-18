import os
import argparse
import logging
import numpy as np
from matplotlib import pyplot as plt

from utils import *
from compressai.models import (MeanScaleHyperprior)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import datasets
from meter import AverageMeter
from ms_ssim_torch import ms_ssim
import wandb

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
logger = logging.getLogger("ImageCompression")
parser = argparse.ArgumentParser(description='Pytorch implementation for (pre-)training the mean-scale hyperprior')
parser.add_argument('--config', dest='config', required=False, help='hyperparameter in json format')
parser.add_argument('--load_json', help='Load settings from json file.')


def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    if global_step < args.lr['warmup_step']:
        lr = args.lr['base'] * global_step / args.lr['warmup_step']
    elif global_step < args.lr['decay_interval']:
        lr = args.lr['base']
    else:
        lr = args.lr['base'] * args.lr['decay']
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(global_step, optimizer, net):
    net.train()

    elapsed, losses, psnrs, bpps, bpp_ys, bpp_zs, mse_losses = [AverageMeter() for _ in range(7)]
    while True:
        if args.train_clic_imagenet:
            logger.info("Training dataset: Combination of ImageNet and CLIC")
            combined_dataset = datasets.get_combined_dataset(path_imagenet=args.path_train["train_imagenet"],
                                                             path_clic=args.path_train["train_clic"],
                                                             image_size=args.image_size,
                                                             cycle_length=args.cycle_length,
                                                             shuffle=True)  # Important for training to shuffle data

            # Load dataset
            train_loader = DataLoader(dataset=combined_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,  # Important for training ImageNet & CLIC to shuffle data
                                      pin_memory=True,
                                      num_workers=0)  # num_workers = 0, else clash with TensorFlow loading

        else:
            logger.info("Training dataset: ImageNet only")
            train_dataset = datasets.get_imagenet_from_tfrecords(args.path_train["train_imagenet"],
                                                                 args.image_size, args.cycle_length)

            # Load dataset
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,  # false for ImageNet only train data, already in TensorFlow loading
                                      pin_memory=True,
                                      num_workers=0)    # num_workers = 0, else clash with TensorFlow loading

        for batch_idx, input in enumerate(train_loader):
            # Adjust learning rate every x steps
            adjust_learning_rate(optimizer, global_step)
            start_time = time.time()

            if global_step >= args.tot_iterations:
                time_since_last = time.time() - start_time
                wandb.log({'train/lr': cur_lr,
                           'train/rd_loss': losses.val,
                           'train/psnr': psnrs.val,
                           'train/bpp': bpps.val,
                           # Measure time for 1 batch (step/sec)
                           'train/steps_per_sec': 1. / time_since_last}, step=global_step)
                wandb.log({'train/bpp_y': bpp_ys.val,
                           'train/bpp_z': bpp_zs.val}, step=global_step)

                process = global_step / args.tot_iterations * 100.0

                log = (' | '.join([
                    f'Step [{global_step}/{args.tot_iterations}={process:.2f}%]',
                    f'Iter {global_step}',
                    f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                    f'Lr {cur_lr}',
                    f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                    f'Bpp_y {bpp_ys.val:.5f} ({bpp_ys.avg:.5f})',
                    f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                    f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
                    f'Step/sec {1. / time_since_last:.5f}',
                ]))
                logger.info(log)
                test_kodak(global_step, net)
                save_model(net, optimizer, global_step, save_path)
                return

            if torch.cuda.is_available():
                input = input.cuda()

            out = net(input)
            x_hat, likelihood_y, likelihood_z = out["x_hat"], out["likelihoods"]["y"], out["likelihoods"]["z"]

            mse_loss = compute_distortion(x_hat, input)
            bpp_y = compute_bpp(likelihood_y, input)
            bpp_z = compute_bpp(likelihood_z, input)
            bpp = bpp_y + bpp_z

            # Multiply with 255**2 to correct for rescaling
            distortion = mse_loss * 255. ** 2

            rd_loss = args.train_lambda * distortion + bpp
            optimizer.zero_grad()
            rd_loss.backward()

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)
            clip_gradient(optimizer, args.lr["clip_grad"])
            optimizer.step()

            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item(), input.size(0))
            else:
                psnrs.update(100, input.size(0))

            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item(), input.size(0))
            bpps.update(bpp.item(), input.size(0))
            bpp_ys.update(bpp_y.item(), input.size(0))
            bpp_zs.update(bpp_z.item(), input.size(0))
            mse_losses.update(mse_loss.item(), input.size(0))
            if (global_step % args.print_freq) == 0:
                # log wandb
                time_since_last = time.time() - start_time
                wandb.log({'train/lr': cur_lr,
                           'train/rd_loss': losses.val,
                           'train/psnr': psnrs.val,
                           'train/bpp': bpps.val,
                           # Measure time for 1 batch (step/sec)
                          'train/steps_per_sec': 1. / time_since_last}, step=global_step)
                wandb.log({'train/bpp_y': bpp_ys.val,
                           'train/bpp_z': bpp_zs.val}, step=global_step)

                process = global_step / args.tot_iterations * 100.0

                log = (' | '.join([
                    f'Step [{global_step}/{args.tot_iterations}={process:.2f}%]',
                    f'Iter {global_step}',
                    f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                    f'Lr {cur_lr}',
                    f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                    f'Bpp_y {bpp_ys.val:.5f} ({bpp_ys.avg:.5f})',
                    f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                    f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
                    f'Step/sec {1. / time_since_last:.5f}',
                ]))
                logger.info(log)

            global_step += 1

            if (global_step % args.save_model_freq) == 0:
                test_kodak(global_step, net)
                save_model(net, optimizer, global_step, save_path)
                net.train()
        logger.info("Finished dataloader, single epoch!")


def test_kodak(step, net):
    logger.info("Test on Kodak dataset: model-{}".format(step))
    with torch.no_grad():
        test_dataset = datasets.TestKodakDataset(data_dir=args.path_test_data)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
        net.eval()
        sum_bpp = 0
        sum_psnr = 0
        sum_msssim = 0
        sum_msssim_db = 0
        sum_rd_loss = 0
        cnt = 0

        for batch_idx, input in enumerate(test_loader):
            if torch.cuda.is_available():
                input = input.cuda()
            out = net(input)
            x_hat, likelihood_y, likelihood_z = out["x_hat"], out["likelihoods"]["y"], out["likelihoods"]["z"]

            mse_loss = compute_distortion(x_hat, input)
            bpp_y = compute_bpp(likelihood_y, input)
            bpp_z = compute_bpp(likelihood_z, input)
            bpp = bpp_y + bpp_z
            clipped_recon_image = x_hat.clamp(0., 1.)

            distortion = mse_loss * 255. ** 2
            rd_loss = args.train_lambda * distortion + bpp

            mse_loss, bpp_y, bpp_z, bpp = torch.mean(mse_loss), torch.mean(bpp_y), torch.mean(bpp_z), torch.mean(bpp)

            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sum_bpp += bpp
            sum_psnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssim_db = -10 * (torch.log(1-msssim) / np.log(10))
            sum_msssim_db += msssim_db
            sum_msssim += msssim
            sum_rd_loss += rd_loss
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, RD Loss: {:.6f}".format(bpp, psnr, msssim, msssim_db, rd_loss))
            cnt += 1

        sum_bpp /= cnt
        sum_psnr /= cnt
        sum_msssim /= cnt
        sum_msssim_db /= cnt
        sum_rd_loss /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, RD Loss: {:.6f}".format(sum_bpp, sum_psnr, sum_msssim, sum_msssim_db, sum_rd_loss))
        # log wandb
        wandb.log({"eval/BPP_Test": sum_bpp,
                   "eval/PSNR_Test": sum_psnr,
                   "eval/MS-SSIM_Test": sum_msssim,
                   "eval/MS-SSIM_DB_Test": sum_msssim_db,
                   "eval/rd_loss": sum_rd_loss}, step=step)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.load_json:
        # Open json file as arguments args
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    cur_lr = args.lr['base']
    global_step = 0

    # Wandb
    wandb.init(project="compression", config=args, name=args.save_name)
    torch.manual_seed(seed=args.seed)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    save_path = os.path.join('checkpoints', args.save_name)
    if args.save_name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("neural image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())

    # Adjust channels for different lambda values
    out_channel_N, out_channel_M = lambda_adjust_channels(args.train_lambda)

    model = MeanScaleHyperprior(N=out_channel_N, M=out_channel_M)

    if torch.cuda.is_available():
        net = model.cuda()
        net = torch.nn.DataParallel(net, list(range(gpu_num)))
    else:
        net = model

    optimizer = optim.Adam(net.parameters(), lr=args.lr['base'])

    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(net, optimizer, args.pretrain)
    logger.info("Global step: {}".format(global_step))

    if args.test:
        test_kodak(global_step, net)
        exit(-1)

    train(global_step, optimizer, net)
