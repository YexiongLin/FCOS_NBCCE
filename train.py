from model.fcos import FCOSDetector
import torch
#from dataset.COCO_dataset import COCODataset
from read_data import *
import math,time
#from dataset.augment import Transforms
import os
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
#transform = Transforms()
train_datas, train_labels=pd.read_csv('trainsets.csv').iloc[:,:].values.T
train_datas, train_labels=get_files(train_datas, train_labels)
train_dataset = liquidDataset(train_datas,train_labels)

model=FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
BATCH_SIZE=opt.batch_size
EPOCHS=opt.epochs
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,
                                         num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))
steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMUP_STEPS=400
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS=0
LR_INIT=0.01
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)
lr_schedule = [800, 1200]
def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

model.train()

for epoch in range(EPOCHS):
    for epoch_step,data in enumerate(train_loader):

        batch_imgs_rgb,batch_imgs_d,batch_boxes,batch_classes=data
        batch_imgs_rgb=batch_imgs_rgb.cuda()
        batch_imgs_d=batch_imgs_d.cuda()
        batch_boxes=batch_boxes.cuda()
        batch_classes=batch_classes.cuda()

        lr = lr_func(GLOBAL_STEPS)
        for param in optimizer.param_groups:
            param['lr']=lr
        
        start_time=time.time()

        optimizer.zero_grad()
        losses=model([batch_imgs_rgb,batch_imgs_d,batch_boxes,batch_classes])
        loss=losses[-1]
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),3)
        optimizer.step()

        end_time=time.time()
        cost_time=int((end_time-start_time)*1000)
        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f nbcce_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f"%\
            (GLOBAL_STEPS,epoch+1,epoch_step+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),losses[3].mean(),cost_time,lr, loss.mean()))


        GLOBAL_STEPS+=1
    
    torch.save(model.state_dict(),"./checkpoint/model_{}.pth".format(epoch+1))
    






