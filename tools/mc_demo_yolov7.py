import argparse
import time
from pathlib import Path
import sys

import cv2, numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from PIL import Image

from tqdm import tqdm
from numpy import random
from MIRNet import MIRNet
sys.path.insert(0, './yolov7')
sys.path.append('.')




from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

def MIRNet_predict(model, bgr_image, device):
    # 確保輸入圖片的大小符合模型的要求
    input_size = (360, 540)
    # img = cv2.resize(bgr_image, input_size)
    cvimg = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    # cvimg.show()
    # 將圖片轉換為PyTorch Tensor，並進行適當的轉換和正規化
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
    ])
    tensor_image = transform(cvimg).unsqueeze(0)
    tensor_image = tensor_image.to(device)

    # 檢查是否使用CUDA
    if torch.cuda.is_available():
        model = model.cuda()  # 確保模型在CUDA上
        tensor_image = tensor_image.cuda()  # 將輸入數據移到CUDA上

    with torch.no_grad():
        output_tensor = model(tensor_image)

    # 將輸出從GPU移回CPU，並移除批次維度
    output_tensor = output_tensor.squeeze(0).cpu().numpy()

    # 將輸出Tensor轉換回PIL圖片，再轉為RGB格式的NumPy陣列
    # output_image = T.ToPILImage()(output_tensor)
    # output_image = np.array(output_image)
    output_img = output_tensor.transpose(1, 2,0)  # 转置轴以适应图像格式 (H, W, C)
    
    # 將RGB轉換回BGR
    # output_bgr_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # 顯示圖片
    # cv2.imshow('im1', output_img)
    # cv2.waitKey(0)
    output_img = cv2.resize(output_img, (1280, 720))
    # cv2.imshow('im1', output_img)
    # cv2.waitKey(0)
    
    return output_img


def img_Enhance(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分離明度通道
    h, s, v = cv2.split(hsv_image)

    # 對明度通道進行直方圖均衡化
    equalized_v = cv2.equalizeHist(v)

    # 合併均衡化後的明度通道和原始色相、飽和度通道
    equalized_hsv_image = cv2.merge([h, s, equalized_v])

    # 將圖像轉換回RGB色彩空間
    enhanced_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image

def canny_invert_add(image):
    # 轉換為灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 進行Canny邊緣檢測
    edges = cv2.Canny(blur, 100, 200)

    green_image = np.zeros_like(image)
    green_image[:] = (0, 255, 0)  # 設置綠色像素值
    
    # 將原始影像中的邊緣部分替換為綠色像素
    green_edges = cv2.bitwise_and(green_image, green_image, mask=edges)
    
    # 將綠色邊緣與原始影像合併
    result = cv2.bitwise_or(image, green_edges)
    return result

def img_EnhanceAndCanny(image):
    # 轉換為灰階影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 對明度通道進行直方圖均衡化
    equalized_img = cv2.equalizeHist(blur)
    # 進行Canny邊緣檢測
    edges = cv2.Canny(equalized_img, 100, 200)
    # 設置綠色像素值
    green_image = np.zeros_like(image)
    green_image[:] = (0, 255, 0)  
    # 將原始影像中的邊緣部分替換為綠色像素
    green_edges = cv2.bitwise_and(green_image, green_image, mask=edges)
    # 將綠色邊緣與原始影像合併
    result = cv2.bitwise_or(image, green_edges)
    return result

def is_nightenv(img):
    arr=[]
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = img[:,:,2]
    arr.append(np.median(v))
    if np.median(v) < 105:
        return True
    else:
        return False
    
def is_nightimg(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # h,s,v = cv2.split(img)
    # k = np.sum(h)/(720*1280)
    # print(np.sum(h))
    # print(int(k),np.mean(h))
    # if k > 30:
    h_channel = img[:, :, 0]
    # 檢查H值是否超過閾值
    average_hue_value = np.mean(h_channel)

    # 檢查平均H值是否超過閾值
    exceeds_threshold = average_hue_value < 60
    # print(f"avg{average_hue_value}")
    return exceeds_threshold
    

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model 
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=1.0)
    tracker_night = BoTSORT(opt, frame_rate=1.0, night_t = True)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
    t0 = time.time()
    
    # Process detections
    results = []
    frameID = 0


    # MIRNET model init
    # modelMIR = MIRNet().to(device)
    # modelMIR.load_state_dict(torch.load(opt.MIRNET_weight))

    for path, img, im0s, vid_cap in tqdm(dataset, desc=f'tracking {opt.name}'):
        frameID += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes
            
            # if is_nightenv(im0):
                # im1 = MIRNet_predict(modelMIR, im0, device)
                # online_targets = tracker.update(detections, im1, frameID)
            # else:
                # online_targets = tracker.update(detections, im0, frameID)
            # im1 = img_EnhanceAndCanny(im0)
            if is_nightimg(im0):
                online_targets = tracker_night.update(detections, im0, frameID)
                # print("SSSSSSSS")
            else:
                online_targets = tracker.update(detections, im0, frameID)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        
                        if 'car' in label: # AICUP only have one cls: car
                            # save results
                            results.append(
                                f"{frameID},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )

                            plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2,score = t.score)

                            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    # a=0
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        with open(save_dir / f"{opt.name}.txt", 'w') as f:
            f.writelines(results)
            
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r"runs\train\TEAM_5038\weights\epoch_049.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold') #default = 0.3 best = 0.3
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS') #default = 0.8
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/example', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.2, help="tracking confidence threshold") #default = 0.3 best = 0.2
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold") #default = 0.05 best=0.05
    parser.add_argument("--new_track_thresh", default=0, type=float, help="new track thresh") #default=0.4
    parser.add_argument("--track_buffer", type=int, default=1, help="the frames for keep lost tracks")#default = 30
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking") #default = 0.7 best = 0.6
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=50, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=True, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default="fast_reid/configs/AICUP/bagtricks_R50-ibn.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"logs\AICUP_115\0509ResNet50_dataProcessed2\model_final.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.4,
                        help='threshold for rejecting low overlap reid matches') #default=0.5 #best=0.8 #best=0.4
    parser.add_argument('--appearance_thresh', type=float, default=0.4,
                        help='threshold for rejecting low appearance similarity reid matches') #default=0.25 #best=0.45 #best=0.4
    
    parser.add_argument('--MIRNET_weight',default="D:\AICUP2024\SpringSeason\Cardetection\pretrain_models\MIRNET\EnhanceMirnet_enhance.pth",type=str)

    parser.add_argument("--fast-reid-nightweights", dest="fast_reid_nightweights", default=r"logs\AICUP_115\0509ResNet50_dataProcessed2\model_final.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_night_thresh', type=float, default=0.8,
                        help='threshold for rejecting low overlap reid matches') #default=0.5 #best=0.8 #best=0.4
    parser.add_argument('--appearance_night_thresh', type=float, default=0.4,
                        help='threshold for rejecting low appearance similarity reid matches') #default=0.25 #best=0.45 #best=0.4
    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
