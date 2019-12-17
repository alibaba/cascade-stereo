from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade Stereo Network (CasStereoNet)')
parser.add_argument('--model', default='gwcnet-c', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--test_dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--test_datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')

parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--ndisps', type=str, default="48,24", help='ndisps')
parser.add_argument('--disp_inter_r', type=str, default="4,1", help='disp_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="32,32,16", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='predicted disp detach, undetach')

parser.add_argument('--using_ns', action='store_true', help='using neighbor search')
parser.add_argument('--ns_size', type=int, default=13, help='nb_size')

parser.add_argument('--test_crop_height', type=int, required=True, help="crop height")
parser.add_argument('--test_crop_width', type=int, required=True, help="crop width")


# parse arguments
args = parser.parse_args()

# dataset, dataloader
Test_StereoDataset = __datasets__[args.test_dataset]
test_dataset = Test_StereoDataset(args.test_datapath, args.testlist, False,
                                  crop_height=args.test_crop_height, crop_width=args.test_crop_width,
                                  test_crop_height=args.test_crop_height, test_crop_width=args.test_crop_width)

TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](
                                maxdisp=args.maxdisp,
                                ndisps=[int(nd) for nd in args.ndisps.split(",") if nd],
                                disp_interval_pixel=[float(d_i) for d_i in args.disp_inter_r.split(",") if d_i],
                                cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                                grad_method=args.grad_method,
                                using_ns=args.using_ns,
                                ns_size=args.ns_size)

model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

model = nn.DataParallel(model)

num_stage = len([int(nd) for nd in args.ndisps.split(",") if nd])

def test():
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join(args.logdir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    outputs = model(sample['left'].cuda(), sample['right'].cuda())
    outputs_stage = outputs["stage{}".format(num_stage)]
    disp_ests = [outputs_stage["pred"]]

    return disp_ests[-1]


if __name__ == '__main__':
    test()
