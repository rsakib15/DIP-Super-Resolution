from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sr import Net as SR
from data import get_training_set
import pdb
import socket
import time

parser = argparse.ArgumentParser(description='Image Super Resulation')
parser.add_argument('--upscale_factor', type=int, default=8, help="SR upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='SR Training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Train learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads')
parser.add_argument('--seed', type=int, default=123, help='Random Seeds')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='training_dataset/')
parser.add_argument('--model_type', type=str, default='SR')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Cropped HR image Size')
#parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth', help='sr pretrained base model')
#parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='SR', help='Location to save checkpoint models')

option = parser.parse_args()
gpus = range(option.gpus)
host = str(socket.gethostname())
cudnn.benchmark = True

print(option)

def train(epoch):
    epoch_loss = 0
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        if cuda:
            input = input.cuda(gpus[0])
            target = target.cuda(gpus[0])
            bicubic = bicubic.cuda(gpus[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if option.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avarage_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        
        if cuda:
            input = input.cuda(gpus[0])
            target = target.cuda(gpus[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avarage_psnr = avg_psnr + psnr

    print("===> Avg. PSNR: {:.4f} dB".format(avarage_psnr / len(testing_data_loader)))


def print_network(net):
    num_params = 0

    for param in net.parameters():
        num_params = num_params + param.numel()
        
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = option.save_folder+host+"_"+option.model_type+option.prefix+"_epoch_{}.pth".format(epoch)
    print("Saving directory is {}".format(option.save_folder))
    if not os.path.exists(option.save_folder):
        os.makedirs(option.save_folder)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {} success!!!!".format(model_out_path))

cuda = option.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(option.seed)
if cuda:
    torch.cuda.manual_seed(option.seed)

print('===> Loading datasets')
train_set = get_training_set(option.data_dir, option.hr_train_dataset, option.upscale_factor, option.patch_size, option.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=option.threads, batch_size=option.batchSize, shuffle=True)

print('===> Building model ', option.model_type)

model = SR(num_channels=3, base_filter=64,  feat=256, num_stages=7, scale_factor=option.upscale_factor)
    
model = torch.nn.DataParallel(model, device_ids=gpus)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

# if option.pretrained:
#     model_name = os.path.join(option.save_folder + option.pretrained_sr)
#     if os.path.exists(model_name):
#         model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
#         print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus[0])
    criterion = criterion.cuda(gpus[0])

optimizer = optim.Adam(model.parameters(), lr=option.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(option.start_iter, option.nEpochs + 1):
    train(epoch)
    if (epoch+1) % (option.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if epoch % (option.snapshots) == 0:
        checkpoint(epoch)