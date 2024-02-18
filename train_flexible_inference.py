import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt  # Without this import compressai cannot be imported
import tensorflow as tf

from compressai.models import (MeanScaleHyperprior)
from baselines_inference import *
from utils import *
from experimental_inference_methods import *

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import json
import time
import datasets
from meter import AverageMeterInference
import logging
import wandb

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
logger = logging.getLogger("ImageCompression")
parser = argparse.ArgumentParser(description='PyTorch implementation for robustly overfitting latents with pre-trained mean-scale hyperprior')
parser.add_argument('--config', dest='config', required=False, help = 'hyperparameter in json format')
parser.add_argument('--load_json', help='Load settings from json file')


def test_inference(z_cur, y_cur, test_net, x):
    with torch.no_grad():
        # Compute z_hat
        test_z_hat = torch.round(z_cur)
        test_likelihood_z = compute_likelihood_z(test_net, test_z_hat)

        # Compute mu_y, sigma_y and y_hat
        test_gaussian_params = test_net.h_s(test_z_hat)
        test_y_sigma, test_y_mu = test_gaussian_params.chunk(2, 1)

        # Fix for irregular image sizes from: https://arxiv.org/abs/2006.04240
        test_y_sigma = test_y_sigma[:, :, :y_cur.size(2), :y_cur.size(3)]
        test_y_mu = test_y_mu[:, :, :y_cur.size(2), :y_cur.size(3)]

        y_cur = y_cur - test_y_mu
        test_y_hat = torch.round(y_cur)
        test_y_hat = test_y_hat + test_y_mu
        test_likelihood_y = compute_likelihood_y(test_net, test_y_hat, test_y_sigma, test_y_mu)

        # Compute x_hat
        test_x_hat = test_net.g_s(test_y_hat)

        # Distortion
        real_mse_loss = compute_distortion(test_x_hat, x)
        real_psnr = 10 * (torch.log(1 / real_mse_loss) / np.log(10))

        # Multiply with 255**2 to correct for rescaling
        real_distortion = real_mse_loss * 255. ** 2

        # Rate
        real_bpp_y = compute_bpp(test_likelihood_y, x)
        real_bpp_z = compute_bpp(test_likelihood_z, x)
        real_bpp = real_bpp_y + real_bpp_z

        # Train with (inference) lambda
        lambda1 = args.train_lambda if 'inference_lambda' not in args.dev_options else args.dev_options['inference_lambda']
        # Rate-distortion loss
        real_rd_loss = lambda1 * real_distortion + real_bpp

    test_dict = {
        "real_psnr": real_psnr,
        "real_mse_loss": real_mse_loss,
        "real_distortion": real_distortion,
        "real_bpp": real_bpp,
        "real_bpp_y": real_bpp_y,
        "real_bpp_z": real_bpp_z,
        "real_rd_loss": real_rd_loss
    }
    return test_dict, test_x_hat


def train_improving_inference(net, global_step):
    # Log metrics
    metric_names = ["elapsed", "losses", "psnrs", "bpps", "bpp_ys", "bpp_zs", "mse_losses",
                    "real_losses", "real_psnrs", "real_bpps", "real_bpp_ys", "real_bpp_zs", "real_mse_losses",
                    "diff_losses"]
    metric_meters = {name: [AverageMeterInference(name) for _ in range(args.tot_inference_iterations)] for name in metric_names}

    # Methods using temperature
    temp_methods = ['danneal', 'sga', 'temp_ste', 'temp_unoise', 'temp_uste', 'temp_gnoise', 'sga_logits', 'sga_logits_3c']

    global_inference_step = 0
    global_batch_i = 0

    while True:
        if 'dev_options' in args and "kodak" in args.dev_options and args.dev_options["kodak"]:
            logger.info("Using KODAK as inference dataset")
            workers = 1 if args.batch_size == 1 else 0
            inference_dataset = datasets.TestKodakDataset(data_dir=args.dev_options["path_test"]["kodak"])
        elif 'dev_options' in args and 'tecnick' in args.dev_options and args.dev_options['tecnick']:
            logger.info("Using TECNICK dataset as inference dataset")
            workers = 1 if args.batch_size == 1 else 0
            inference_dataset = datasets.TestTecnickDataset(data_dir=args.dev_options["path_test"]["tecnick"])
        elif 'dev_options' in args and "pathology" in args.dev_options and args.dev_options["pathology"]:
            logger.info("Using PATHOLOGY dataset 'nuclei' as inference dataset")
            workers = 1 if args.batch_size == 1 else 0
            inference_dataset = datasets.TestPathologyDataset(data_dir=args.dev_options["path_test"]["pathology"])
        else:
            logger.info("Using ImageNet as inference training dataset")
            workers = 0
            inference_dataset = datasets.get_imagenet_from_tfrecords(args.path_train["train_imagenet"],
                                                                 args.image_size,
                                                                 cycle_length=args.cycle_length,
                                                                 shuffle=False)

        # Load dataset
        inference_loader = DataLoader(dataset=inference_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=workers)  # num_workers = 0, else clash with TensorFlow loading

        net.eval()
        for batch_idx, input in enumerate(inference_loader):
            start_time = time.time()
            global_step += 1
            global_batch_i += 1

            if global_batch_i > args.tot_batches:
                # Compute the averages for each metric
                averages = {}
                for metric_name, metric_meter_list in metric_meters.items():
                    metric_averages = []
                    if len(metric_meter_list) != args.tot_inference_iterations:
                        raise ValueError(f"Length list {len(metric_meter_list)} does not match num iter: {args.tot_inference_iterations}")
                    for metric_meter in metric_meter_list:
                        if metric_meter.sum == 0 and not "diff_losses":
                            raise NotImplementedError(f"Watch out for this!!! Metric: {metric_name}")
                        avg = metric_meter.sum / metric_meter.count
                        metric_averages.append(avg)
                    averages[metric_name] = metric_averages

                # Returns the average for each iteration over the different batches
                for i in range(args.tot_inference_iterations):
                    wandb.log({'train/compressai_inference/rd_loss': averages['losses'][i],
                               'train/compressai_inference/psnr': averages['psnrs'][i],
                               'train/compressai_inference/bpp': averages['bpps'][i],
                               'train/compressai_inference/real_rd_loss': averages['real_losses'][i],
                               'train/compressai_inference/real_psnr': averages['real_psnrs'][i],
                               'train/compressai_inference/real_bpp': averages['real_bpps'][i],
                               'train/compressai_inference/diff_loss': averages['diff_losses'][i],
                               'train/compressai_inference/inference_step': i}, step=i)
                    wandb.log({'train/compressai_inference/bpp_y': averages['bpp_ys'][i],
                               'train/compressai_inference/bpp_z': averages['bpp_zs'][i],
                               'train/compressai_inference/real_bpp_y': averages['real_bpp_ys'][i],
                               'train/compressai_inference/real_bpp_z': averages['real_bpp_zs'][i],
                               'train/compressai_inference/inference_step': i}, step=i)

                    if bottleneck_method in temp_methods:
                        if "dev_options" in args and "improving_methods" in args.dev_options:
                            temp = bottleneck.temp_method.annealed_temperature(i)
                            if bottleneck_method == "sga_logits" or bottleneck_method == "sga_logits_3c":
                                temp = 1. - temp
                        else:
                            temp = annealed_temperature(i)
                        wandb.log({'train/compressai_inference/temperature': temp}, step=i)

                logger.info("Done training! Ended at global batch: {} | number of inference step per batch: {} | "
                            "global inference steps: {}".format(global_batch_i, args.tot_inference_iterations, global_inference_step))
                return

            if torch.cuda.is_available():
                input = input.cuda()

            logger.info("Global step: {}, Global batch: {}".format(global_inference_step, global_batch_i))

            # Start training to improve flexible inference
            # initialize and detach: y, z
            y_cur = net.g_a(input)
            z_cur = net.h_a(y_cur)

            y_cur = y_cur.clone().detach().requires_grad_(True)
            z_cur = z_cur.clone().detach().requires_grad_(True)

            if "dev_options" in args and "custom_lr" in args.dev_options:
                learning_rate = args.dev_options['custom_lr']['start_lr']
            else:
                learning_rate = args.optimizer["lr"]

            optimizer = optimizer_choice(args.optimizer["name"], y_cur, z_cur, learning_rate, args.optimizer["momentum"])

            lambda1 = args.train_lambda if 'inference_lambda' not in args.dev_options else args.dev_options['inference_lambda']

            logger.info(f"Optimizer: {args.optimizer['name']} with learning rate: {learning_rate} "
                        f"(optional momentum: {args.optimizer['momentum']}) | lambda: {lambda1}")

            if "dev_options" in args and "improving_methods" in args.dev_options:
                bottleneck_method = next(key for key, value in args.dev_options['improving_methods'].items() if value)
                rate = args.dev_options['temp_rate']
                if bottleneck_method == "sga_logits" or bottleneck_method == "sga_logits_3c":
                    sga_params = args.dev_options['sga_logits_params']
                    ub = sga_params['ub']
                    temp_method_class = Temperature(method=bottleneck_method, rate=rate, ub=ub)
                else:
                    temp_method_class = Temperature(method=bottleneck_method, rate=rate)
                instance = {'temp_method_class': temp_method_class}

                if bottleneck_method == "gnoise":
                    instance['gnoise_param'] = args.dev_options['gnoise_sigma']
                if bottleneck_method == "sga_logits":
                    instance['logits'] = sga_params['logits']
                    instance['scale_factor'] = args.dev_options['scale_factor']
                if bottleneck_method == "sga_logits_3c":
                    instance['logits'] = sga_params['logits']
                    instance['scale_factor'] = args.dev_options['scale_factor']
                    instance['3c_power'] = args.dev_options['3c_power']
                    instance['3c_factor'] = args.dev_options['3c_factor']

                bottleneck = ExperimentalBottleneck(args.dev_options['improving_methods'], **instance)

                if global_inference_step == 0:
                    logger.info(f"Method: {bottleneck_method} | temperature rate = {rate}")
                    if bottleneck_method == "gnoise":
                        logger.info(f"mu: 0 & sigma: {args.dev_options['gnoise_sigma']}")
                    if bottleneck_method == "sga_logits" or bottleneck_method == "sga_logits_3c":
                        logger.info(f"SGA+ logits choice: {sga_params['logits']} | ub: {ub}")
            else:
                logger.info('Using the default bottleneck (necessary for STE and unoise')
                bottleneck = Bottleneck(args.improving_methods)
                bottleneck_method = bottleneck.method

            for iteration in range(args.tot_inference_iterations):
                global_inference_step += 1
                if "dev_options" in args and "custom_lr" in args.dev_options:
                    custom_learning_rate(optimizer, iteration, args.dev_options['custom_lr'])

                # Compute zhat
                z_hat = bottleneck.computing_method(z_cur, y_cur, iteration, compute_zhat=True)
                # Compute entropy zhat
                likelihood_z = compute_likelihood_z(net, z_hat)

                # Compute yhat
                gaussian_params = net.h_s(z_hat)
                y_sigma, y_mu = gaussian_params.chunk(2, 1)

                # Fix for irregular image sizes from: https://arxiv.org/abs/2006.04240
                y_sigma = y_sigma[:, :, :y_cur.size(2), :y_cur.size(3)]
                y_mu = y_mu[:, :, :y_cur.size(2), :y_cur.size(3)]

                y_hat = bottleneck.computing_method(z_cur, y_cur, iteration, compute_zhat=False, mu=y_mu)

                # Compute entropy yhat
                likelihood_y = compute_likelihood_y(net, y_hat, scales=y_sigma, means=y_mu)

                # Reconstructed image xhat
                x_hat = net.g_s(y_hat)

                # Distortion
                mse_loss = compute_distortion(x_hat, input)

                # Multiply with 255**2 to correct for rescaling
                distortion = mse_loss * 255. ** 2

                # Rate
                bpp_y = compute_bpp(likelihood_y, input)
                bpp_z = compute_bpp(likelihood_z, input)
                bpp = bpp_y + bpp_z

                # Rate-distortion loss
                # Train with (inference) lambda
                rd_loss = lambda1 * distortion + bpp

                real_dict, test_x_hat = test_inference(z_cur, y_cur, net, input)

                # Compute difference in losses
                real_loss = real_dict["real_rd_loss"]
                diff_loss = rd_loss - real_loss

                if "dev_options" in args and "augmented_loss" in args.dev_options and args.dev_options["augmented_loss"]:
                    if global_inference_step == 1:
                        logger.info("Training on augmented loss!!!")
                    loss = (rd_loss + real_loss) * .5
                    # loss = rd_loss + diff_loss**2
                else:
                    loss = rd_loss

                # Gradients for backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the corresponding loss_meter with the computed value
                metric_meters["elapsed"][iteration].update(time.time() - start_time)
                metric_meters["losses"][iteration].update(rd_loss.item(), input.size(0))
                metric_meters["mse_losses"][iteration].update(mse_loss.item(), input.size(0))
                metric_meters["bpps"][iteration].update(bpp.item(), input.size(0))

                # Update real improvements
                metric_meters["real_bpps"][iteration].update(real_dict["real_bpp"].item(), input.size(0))
                metric_meters["real_losses"][iteration].update(real_dict["real_rd_loss"].item(), input.size(0))
                metric_meters["real_mse_losses"][iteration].update(real_dict["real_mse_loss"].item(), input.size(0))

                # Update loss difference
                metric_meters["diff_losses"][iteration].update(diff_loss.item(), input.size(0))

                if mse_loss.item() > 0:
                    psnr = 10 * (torch.log(1 / mse_loss) / np.log(10))
                    metric_meters["psnrs"][iteration].update(psnr.item(), input.size(0))
                    metric_meters["real_psnrs"][iteration].update(real_dict["real_psnr"].item(), input.size(0))
                else:
                    psnr = 0
                    metric_meters["psnrs"][iteration].update(psnr, input.size(0))
                    metric_meters["real_psnrs"][iteration].update(psnr, input.size(0))

                if "dev_options" in args and "save_final" in args.dev_options and args.dev_options['save_final']:
                    if (iteration == 0) or (iteration == 499) or (iteration == args.tot_inference_iterations - 1):
                        logger.info(f"Save images: batch [{global_batch_i}] | iter [{iteration}]")
                        round_xhat_img = test_x_hat[0]
                        round_xhat_caption = (f"true_xhat_batch{global_batch_i}_iter{iteration}"
                                                f"_loss_{round(real_loss.item(), 4)}"
                                                f"_BPP_{round(real_dict['real_bpp'].item(), 4)}"
                                                f"_PSNR_{round(real_dict['real_psnr'].item(), 4)}")

                        save_dir = os.path.join(save_path, "visual_results")
                        os.makedirs(save_dir, exist_ok=True)
                        save_image(round_xhat_img, os.path.join(save_dir, f"{round_xhat_caption}.png"))

                metric_meters["bpp_ys"][iteration].update(bpp_y.item(), input.size(0))
                metric_meters["bpp_zs"][iteration].update(bpp_z.item(), input.size(0))
                metric_meters["real_bpp_ys"][iteration].update(real_dict["real_bpp_y"].item(), input.size(0))
                metric_meters["real_bpp_zs"][iteration].update(real_dict["real_bpp_z"].item(), input.size(0))

                if (iteration % args.print_freq) == 0:
                    time_since_last = time.time() - start_time
                    total_steps = int(args.tot_batches) * int(args.tot_inference_iterations)
                    process = (global_inference_step / total_steps) * 100

                    log = (' | '.join([
                        f'Batch [{batch_idx}] | Iter [{iteration}]',
                        f'Step [{global_inference_step}/{total_steps}={process:.2f}%]',
                        f'Time {metric_meters["elapsed"][iteration].val:.3f} ({metric_meters["elapsed"][iteration].avg:.3f})',
                        f'\nTotal Loss {metric_meters["losses"][iteration].val:.3f} ({metric_meters["losses"][iteration].avg:.3f})',
                        f'PSNR {metric_meters["psnrs"][iteration].val:.3f} ({metric_meters["psnrs"][iteration].avg:.3f})',
                        f'Bpp {metric_meters["bpps"][iteration].val:.5f} ({metric_meters["bpps"][iteration].avg:.5f})',
                        f'Bpp_y {metric_meters["bpp_ys"][iteration].val:.5f} ({metric_meters["bpp_ys"][iteration].avg:.5f})',
                        f'Bpp_z {metric_meters["bpp_zs"][iteration].val:.5f} ({metric_meters["bpp_zs"][iteration].avg:.5f})',
                        f'MSE {metric_meters["mse_losses"][iteration].val:.5f} ({metric_meters["mse_losses"][iteration].avg:.5f})',
                        f'\nTotal real_Loss {metric_meters["real_losses"][iteration].val:.3f} ({metric_meters["real_losses"][iteration].avg:.3f})',
                        f'real_PSNR {metric_meters["real_psnrs"][iteration].val:.3f} ({metric_meters["real_psnrs"][iteration].avg:.3f})',
                        f'real_Bpp {metric_meters["real_bpps"][iteration].val:.5f} ({metric_meters["real_bpps"][iteration].avg:.5f})',
                        f'real_Bpp_y {metric_meters["real_bpp_ys"][iteration].val:.5f} ({metric_meters["real_bpp_ys"][iteration].avg:.5f})',
                        f'real_Bpp_z {metric_meters["real_bpp_zs"][iteration].val:.5f} ({metric_meters["real_bpp_zs"][iteration].avg:.5f})',
                        f'real_MSE {metric_meters["real_mse_losses"][iteration].val:.5f} ({metric_meters["real_mse_losses"][iteration].avg:.5f})',
                        f'real_distortion {real_dict["real_distortion"].item():.5f}',
                        f'Loss difference: {diff_loss.item():.5f}',
                        f'Step/sec {1. / time_since_last:.5f}',
                    ]))
                    logger.info(log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.load_json:
        # Open json file as arguments args
        with open(args.config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    global_step = 0

    # Write configs to wandb
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
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())

    out_channel_N, out_channel_M = lambda_adjust_channels(args.train_lambda)

    model = MeanScaleHyperprior(N=out_channel_N, M=out_channel_M)

    if args.pretrain != '':
        global_step = load_model_inference(model, None, args.pretrain)
        logger.info("Pre-trained model from global step: {}".format(global_step))
    else:
        raise ValueError("Please use a pre-trained model to improve inference!!!")

    if torch.cuda.is_available():
        net = model.cuda()
        net = torch.nn.DataParallel(net, list(range(gpu_num)))
    else:
        net = model

    net = net.module if isinstance(net, torch.nn.DataParallel) else net

    train_improving_inference(net, global_step)

