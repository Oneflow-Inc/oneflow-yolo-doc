## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>


源码解读： [detect.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/detect.py)

> Run inference with a YOLOv5 model on images, videos, directories, streams

## 1. 导入需要的包和基本配置


```python
import argparse  # python的命令行解析的标准模块  可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os #  os库是Python标准库,包含几百个函数,常用路径操作、进程管理、环境参数等几类。
import platform # 用platform模块可以判断当前的系统环境
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np # Numpy是使用C语言实现的一个数据计算库
import oneflow as flow # OneFlow框架
import oneflow.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box
from utils.oneflow_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
```

## 2. opt参数详解


| 参数           | 解析                                             |                                                                                                                           |
|----------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| weights        | model path(s)                                    | 模型的权重地址                                                                                                            |
| source         | file/dir/URL/glob, 0 for webcam                  | 测试数据文件(图片或视频)的保存路径 默认data/images                                                                        |
| data           | (optional) dataset.yaml path                     | 数据集配置文件路径                                                                                                        |
| imgsz          | inference size h,w                               | 网络输入图片的大小 默认640                                                                                                |
| conf-thres     | confidence threshold                             | object置信度阈值 默认0.25                                                                                                 |
| iou-thres      | NMS IoU threshold                                | 做nms的iou阈值 默认0.45                                                                                                   |
| max-det        | maximum detections per image                     | 每张图片最大的目标个数 默认1000                                                                                           |
| device         | cuda device, i.e. 0 or 0,1,2,3 or cpu            | 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu                                                                  |
| view-img       | show results                                     | 是否展示预测之后的图片或视频 默认False                                                                                    |
| save-txt       | save results to *.txt                            | 是否将预测的框坐标以txt文件格式保存 默认False 会在runs/detect/expn/labels下生成每张图片预测的txt文件                      |
| save-conf      | save confidences in --save-txt labels            | 是否保存预测每个目标的置信度到预测tx文件中 默认False                                                                      |
| save-crop      | save cropped prediction boxes                    | 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False |
| nosave         | do not save images/videos                        | 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片                                                            |
| classes        | filter by class: --classes 0, or --classes 0 2 3 | 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留                                               |
| agnostic-nms   | class-agnostic NMS                               | 进行nms是否也除去不同类别之间的框 默认False                                                                               |
| augment        | augmented inference                              | 预测是否也要采用数据增强 TTA                                                                                              |
| visualize      | visualize features                               | 可视化特征                                                                                                                |
| update         | update all models                                | 是否将optimizer从ckpt中删除  更新模型  默认False                                                                          |
| project        | save results to project/name                     | 当前测试结果放在哪个主文件夹下 默认runs/detect                                                                            |
| name           | save results to project/name                     | 当前测试结果放在run/detect下的文件名  默认是exp                                                                           |
| exist-ok       | existing project/name ok, do not increment       | 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹                                            |
| line-thickness | bounding box thickness (pixels)                  | 画框的框框的线宽  默认是 3                                                                                                |
| hide-labels    | hide labels                                      | 画出的框框是否需要隐藏label信息 默认False                                                                                 |
| hide-conf      | hide confidences                                 | 画出的框框是否需要隐藏label信息 默认False                                                                                 |
| half           | use FP16 half-precision inference                | 画出的框框是否需要隐藏conf信息 默认False                                                                                  |
| dnn            | use OpenCV DNN for ONNX inference                | 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False                                                              |

## 3 main函数


```python
def main(opt):
    # 检查包是否满足requirements对应txt文件的要求
    check_requirements(exclude=('tensorboard', 'thop'))
    # 执行run 开始推理
    run(**vars(opt))
```

## 4 run函数


```python

@flow.no_grad()
def run(
    weights=ROOT / "yolov5s",  # model path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    # 文件类型
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 是否是url网络地址
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    # 是否是使用webcam 网页数据 一般是Fasle  因为我们一般是使用图片流LoadImages(可以处理图片/视频流文件)
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file: 
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增量运行
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建文件夹用于存储输出结果

    # Load model
    device = select_device(device) # 获取当前主机可用的设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # stride: 模型最大的下采样率 [8, 16, 32] 所有stride一般为32
    # names: 得到数据集的所有类的类名
    # of : oneflow模型权重文件
    stride, names, of = model.stride, model.names, model.of
    
    #  确保输入图片的尺寸imgsz能整除stride  如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:  # 一般不会使用webcam模式从网页中获取数据
        view_img = check_imshow()
        cudnn.benchmark = True  # 设置为True，使用CUDNN以加快恒定图像尺寸推断的速度
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=of)
        bs = len(dataset)  # batch_size
    else: # 一般是直接从source文件目录下直接读取图片或者视频数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=of)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 正式推理
    model.warmup(imgsz=(1 if of else bs, 3, *imgsz))  # warmup

    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = flow.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 去掉detection任务重复的检测框。跟多请参阅 https://blog.csdn.net/yql_617540298/article/details/89474226
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image  对每张图片进行处理  将pred(相对img_size 640)映射回原图img0 size
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = flow.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息（相对img_size 640）映射回原图 img0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                det = det.detach().cpu().numpy()

                # Print results
                # 输出信息s + 检测到的各个类别的目标个数
                for c in np.unique(det[:, -1]):
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(flow.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    
                    # 在原图上画框 + 将预测到的目标剪切出来 
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:# 如果需要就将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                        save_one_box(
                            xyxy,
                            imc,
                            file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                            BGR=True,
                        )

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img: # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}Done. ({t3 - t2:.3f}s)")

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"%.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # strip_optimizer函数将optimizer从ckpt中删除  更新模型
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
```
