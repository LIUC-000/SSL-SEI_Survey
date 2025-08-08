import pdb
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imp
import numpy as np
import random
import torch
import torchvision

import config_env_example as env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
RANDOM_SEED = 300  # any random number
set_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='Rot_Predict_CVCNN',help='config file with parameters of the experiment')
parser.add_argument('--evaluate', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--disp_step', type=int, default=50, help='display step during training')
parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint (epoch id) that will be loaded')
args_opt = parser.parse_args()

exp_config_file = os.path.join('.', 'config', args_opt.exp + '.py')
# if args_opt.semi == -1:
exp_directory = os.path.join("_experiments", args_opt.exp)
# else:
#    assert(args_opt.semi>0)
#    exp_directory = os.path.join('.','experiments/unsupervised',args_opt.exp+'_semi'+str(args_opt.semi))

# 加载实验的配置参数
logging.info('Launching experiment: %s' % exp_config_file)
config = imp.load_source("", exp_config_file).config
config['exp_dir'] = exp_directory  # 存储日志、模型和其他内容的地方
logging.info("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
logging.info("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# 设置训练和测试数据集以及相应的数据加载器
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

config['disp_step'] = args_opt.disp_step

from dataloader import get_database, GenericDataset, GenericDataLoader

wifi_database = get_database(env.WIFI_DIR)

dataset_train = GenericDataset(
    database=wifi_database,
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'])
dataset_test = GenericDataset(
    database=wifi_database,
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'])

dloader_train = GenericDataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    num_workers=args_opt.num_workers,
    shuffle=True)
dloader_test = GenericDataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    num_workers=args_opt.num_workers,
    shuffle=False)

is_evaluation = True if (args_opt.evaluate == 1) else False
if is_evaluation:
    logging.info("### ----- Evaluation: inference only. ----- ###")
else:
    logging.info("### ----- Training: train model. ----- ###")

if torch.cuda.is_available():
    logging.info("### ----- GPU device available, arrays will be copied to cuda. ----- ###")
else:
    logging.info("### ----- GPU device is unavailable, computation will be performed on CPU. ----- ###")

import algorithms as alg

algorithm = getattr(alg, config['algorithm_type'])(config)

if torch.cuda.is_available():  # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint > 0:  # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train=(not is_evaluation))

if not is_evaluation:  # train the algorithm
    algorithm.solve(dloader_train, dloader_test)
else:
    algorithm.evaluate(dloader_test)  # evaluate the algorithm
