import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from albumentations import *
from albumentations.pytorch import ToTensor


def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

if __name__=="__main__":
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=10
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32]
        limit_range=[[-1,150],[150,300],[300,999999]]

        #inference
        score_threshold=0.4
        nms_iou_threshold=0.6
        max_detection_boxes_num=300

        theta=0.5

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_38.pth",map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.eval()
    print("===>success loading model")
    depth_transform = Compose([Normalize((414.76987), (171.60141), always_apply=True),
                                         ToTensor()])
    import os
    root="./test_images/RGB/"
    names=os.listdir(root)
    for name in names:
        print(root+name)
        img_bgr=cv2.imread(root+name)
        depth=cv2.imread((root+name).replace('RGB','depth'),-1)
        #img_pad=preprocess_img(img_bgr,[480,640])
        img_pad=img_bgr
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img0=transforms.ToTensor()(img)
        img0= transforms.Normalize([0.48877397, 0.47886574, 0.45613098], [0.04925239, 0.06920359, 0.10496992],inplace=True)(img0)
        print(img0.shape)
        #depth=preprocess_img(depth,[480,640])
        img1=depth_transform(image=depth)['image'].T
        img1=img1.unsqueeze(0)
        
        print(img1.shape)
        '''
        img1=np.asarray(depth,np.int16)
        img1=transforms.ToTensor()(img1)
        img1= transforms.Normalize([414.76987], [171.60141],inplace=True)(img1)
        '''
        

        start_t=time.time()
        with torch.no_grad():
            out=model([img0.unsqueeze_(dim=0),img1.unsqueeze_(dim=0)])
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out
        print(scores)
        print(classes)
        print(boxes)

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            b_color = colors[int(classes[i]/46) - 1]
            bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)
            ax.add_patch(bbox)
            plt.text(box[0], box[1], s="%s %.3f"%(str(classes[i]),scores[i]), color='white',
                     verticalalignment='top',
                     bbox={'color': b_color, 'pad': 0})
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('out_images/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        plt.close()





