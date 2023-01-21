import pyrealsense2 as rs
import numpy as np
import cv2
import os
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from albumentations import *
from albumentations.pytorch import ToTensor


def fun(im):
	im=np.asarray(im,np.uint16)
	return im

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
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_38.pth",map_location=torch.device('cpu')))
    model=model.eval()
    print("===>success loading model")
    depth_transform = Compose([Normalize((414.76987), (171.60141), always_apply=True),
                                         ToTensor()])

    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  #10、15或者30可选,20或者25会报错，其他帧率未尝试
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    cnt = 14
    try:
        while True:
            frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            if not aligned_depth_frame:
                continue
            depth_frame = np.asanyarray(aligned_depth_frame.get_data(),dtype='uint16')
            cv2.imshow('1 depth', depth_frame/1024.)

        # color frames
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            color_frame = np.asanyarray(color_frame.get_data())
            rgb_img=color_frame.copy()
            cv2.imshow('2 color', color_frame)

            img0=transforms.ToTensor()(color_frame)
            img0= transforms.Normalize([0.48877397, 0.47886574, 0.45613098], [0.04925239, 0.06920359, 0.10496992],inplace=True)(img0)

            img1=depth_transform(image=depth_frame)['image'].T
            img1=img1.unsqueeze(0)

            start_t=time.time()
            with torch.no_grad():
                out=model([img0.unsqueeze_(dim=0),img1.unsqueeze_(dim=0)])
            end_t=time.time()
            cost_t=1000*(end_t-start_t)
            print("===>success processing img, cost time %.2f ms"%cost_t)
            # print(out)
            scores,classes,boxes=out

            boxes=boxes[0].cpu().numpy().tolist()
            classes=classes[0].cpu().numpy().tolist()
            scores=scores[0].cpu().numpy().tolist()

            for i,box in enumerate(boxes):
                pt1=(int(box[0]),int(box[1]))
                pt2=(int(box[2]),int(box[3]))
                cv2.rectangle(color_frame,pt1,pt2,(0,0,255.),2)
                cv2.putText(color_frame,"%s %.3f"%(str(classes[i]),scores[i]),(int(box[0]-15), int(box[1]) - 4),0, 0.5,[255, 255, 255],thickness=1,lineType=cv2.LINE_AA)

            cv2.imshow('results', color_frame)

            if os.path.exists('results')==False:
                os.mkdir('results')
            if os.path.exists('depth')==False:
                os.mkdir('depth')
            if os.path.exists('RGB')==False:
                os.mkdir('RGB')

            key = cv2.waitKey(1)
            if key == ord('t'):
                #print(cnt)
                cv2.imwrite('./results/' + str(cnt) + '.png', color_frame)
                cv2.imwrite('./depth/' + str(cnt) + '.png', fun(depth_frame))
                cv2.imwrite('./RGB/' + str(cnt) + '.png', rgb_img)
                #cnt += 1
            if key == ord("q") or key == 27:
                break


    finally:
        # Stop streaming
        pipeline.stop()
