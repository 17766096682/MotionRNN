__author__ = 'yunbo'

import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer1 as trainer

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--data_train_path', type=str, default='../data/')
parser.add_argument('--data_val_path', type=str, default='../data/')
parser.add_argument('--data_test_path', type=str, default='../data/')
parser.add_argument('--save_dir', type=str, default='checkpoints/radar_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/radar_predrnn')
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--total_length', type=int, default=15)
parser.add_argument('--img_height', type=int, default=128)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--model_name', type=str, default='MotionRNN_PredRNN')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
# ----------------------------------------------------------------------------
parser.add_argument('--patch_size', type=int, default=4)
# ----------------------------------------------------------------------------

parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)

parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=10)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=5)
# parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()


# print(args)


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model):
    begin = 0
    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])
    # load data
    train_input_handle = datasets_factory.data_provider(dataset=args.dataset_name,
                                                        configs=args,
                                                        data_train_path=args.data_train_path,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)

    test_input_handle = datasets_factory.data_provider(dataset=args.dataset_name,
                                                       configs=args,
                                                       data_train_path=args.data_train_path,
                                                       data_test_path=args.data_val_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)  # 训练到5000次后为0
    itr = begin
    for epoch in range(0, args.max_iterations):
        if itr > args.max_iterations:
            break

        for ims in train_input_handle:
            if itr > args.max_iterations:
                break
            ims = preprocess.reshape_patch(ims, args.patch_size)

            eta, real_input_flag = schedule_sampling(eta, itr)

            if itr == 0:
                print("Validate")
                trainer.test(model, test_input_handle, args, itr)

            trainer.train(model, ims, real_input_flag, args, itr)

            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)

            if itr % args.test_interval == 0 and itr != 0:
                print("Validata")
                trainer.test(model, test_input_handle, args, itr)

            itr += 1


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(dataset=args.dataset_name,
                                                       configs=args,
                                                       data_train_path=args.train_data_paths,
                                                       data_test_path=args.valid_data_paths,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)
    trainer.test(model, test_input_handle, args, 'test_result')


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

# gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
# args.n_gpu = len(gpu_list)
print('Initializing models')

model = Model(args)
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.network.parameters()) / 1000000.0))

if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
