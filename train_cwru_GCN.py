#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, argparse, logging
from datetime import datetime
from utils.logger import setlogger
from utils.train_utils_dong_GCN import train_utils


os.environ['CUDA_LAUNCH_BLOCKING']='1'
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--label_type', type=str, choices=['3_25.2', '0_0', '1_1.4', '2_2.8'], default='3_25.2',
                        help='Working Condition of the data')#没用
    parser.add_argument('--model_name', type=str, default='GCN', help='the name of the model_name')
    parser.add_argument('--pretrained_model_name', type=str, default='densenet121',
                        help='the name of the pretrained torch_model_name')
    parser.add_argument('--data_name', type=str, default='CWRU_RGCN', help='the name of the data')  # 数据预处理方式(raopt)
    parser.add_argument('--data_dir', type=str, default="./label/DONG_label",
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A', 'custom'], default='custom',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, '
                             'O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='the directory to save the model_name')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model_name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam', 'custom'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate, custom is made by raopt')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix', 'dan'], default='stepLR',
                        help='the learning rate schedule, dan is made by raopt based on the dan train code')
    parser.add_argument('--gamma', type=float, default=0.9, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='10', help='the learning rate decay for step and stepLR')
    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=10, help='max number of epoch')
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
    trainer.setup_seed()
    trainer.setup()
    trainer.train()
