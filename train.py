#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, argparse, logging
from datetime import datetime
from utils.logger import setlogger
from utils.train_utils import train_utils
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='Inception2d', help='the name of the model_name')
    parser.add_argument('--data_name', type=str, default='CWRUSlice', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default="E:\Data\西储大学轴承数据中心网站",
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, '
                             'O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model_name')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model_name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model_name
    sub_dir = args.model_name+'_'+args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()





