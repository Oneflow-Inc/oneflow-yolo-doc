## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>


源码解读： [val.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/val.py)

Ultralytics YOLOv5 官方给的介绍:

> Validate a model's accuracy on [COCO](https://cocodataset.org/#home) val or test-dev datasets. Models are downloaded automatically from the [latest YOLOv5 release](https://github.com/Oneflow-Inc/one-yolov5/releases). To show results by class use the `--verbose` flag. Note that `pycocotools` metrics may be ~1% better than the equivalent repo metrics, as is visible below, due to slight differences in mAP computation.



## 1.导入需要的包和基本配置


```python
import argparse # 解析命令行参数模块
import json     # 字典列表和JSON字符串之间的相互解析模块
import os       # 与操作系统进行交互的模块 包含文件路径操作和解析
import sys      # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np # NumPy（Numerical Python）是Python的一种开源的数值计算扩展
import oneflow as flow # OneFlow 深度学习框架
from tqdm import tqdm # 进度条模块
 
from models.common import DetectMultiBackend # 下面都是 one-yolov5 定义的模块，在本系列的其它文章都有涉及
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.oneflow_utils import select_device, time_sync
from utils.plots import output_to_target, plot_images, plot_val_study
```

## 2.opt参数详解

| 参数        | 解析                                                                                            |                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| data        | dataset.yaml path                                                                               | 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 |
| weights     | model weights path(s)                                                                           | 模型的权重文件地址 weights/yolov5s                                  |
| batch-size  | batch size                                                                                      | 计算样本的批次大小 默认32                                           |
| imgsz       | inference size (pixels)                                                                         | 输入网络的图片分辨率    默认640                                     |
| conf-thres  | confidence threshold                                                                            | object置信度阈值 默认0.001                                          |
| iou-thres   | NMS IoU threshold                                                                               | 进行NMS时IOU的阈值 默认0.6                                          |
| task        | train, val, test, speed or study                                                                | 设置测试的类型 有train, val, test, speed or study几种 默认val       |
| device      | cuda device, i.e. 0 or 0,1,2,3 or cpu                                                           | 测试的设备                                                          |
| workers     | max dataloader workers (per RANK in DDP mode)                                                   | 加载数据使用的 dataloader workers                                   |
| single-cls  | treat as single-class dataset                                                                   | 数据集是否只用一个类别 默认False                                    |
| augment     | [augmented inference](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/TTA.html) | 测试是否使用TTA Test Time Augment 默认False                         |
| verbose     | report mAP by class                                                                             | 是否打印出每个类别的mAP 默认False                                   |
| save-hybrid | save label+prediction hybrid results to *.txt                                                   | 保存label+prediction 杂交结果到对应.txt 默认False                    |
| save-conf   | save confidences in --save-txt labels                                                           |                                                                     |
| save-json   | save a COCO-JSON results file                                                                   | 是否按照coco的json格式保存结果       默认False                      |
| project     | save to project/name                                                                            | 测试保存的源文件 默认`runs/val`                                     |
| name        | save to project/name                                                                            | 测试保存的文件地址名 默认`exp`  保存在`runs/val/exp`下              |
| exist-ok    | existing project/name ok, do not increment                                                      | 是否保存在当前文件，不新增 默认False                                            |
| half        | use FP16 half-precision inference                                                               | 是否使用半精度推理 默认False                                        |
| dnn         | use OpenCV DNN for ONNX inference                                                               | 是否使用 `OpenCV DNN` 对 `ONNX` 模型推理                              |

## 3.[main函数](https://github.com/Oneflow-Inc/one-yolov5/blob/bf8c66e011fcf5b8885068074ffc6b56c113a20c/val.py#L443)

> 根据解析的opt参数，调用run函数


```python
def main(opt):
    #  检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # 更多请见 https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml
            #                --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml
            #                --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"
                x, y = (
                    list(range(256, 1536 + 128, 128)),
                    [],
                )  # x axis (image sizes), y axis
                # "study": 模型在各个尺度下的指标并可视化，
                # 上面list(range(256, 1536 + 128, 128)),代表 img-size 的各个尺度, 具体代码如下：
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            os.system("zip -r study.zip study_*.txt")
            # 可视化各个指标
            plot_val_study(x=x)  # plot
```

## 3. run函数
> https://github.com/Oneflow-Inc/one-yolov5/blob/bf8c66e011fcf5b8885068074ffc6b56c113a20c/val.py#L112-L383

### 3.1 载入参数


```python
# 不参与反向传播
@flow.no_grad() 
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",   # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,    # use FP16 half-precision inference
    dnn=False,    # use OpenCV DNN for ONNX inference
    model=None,   # 模型 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
    dataloader=None,   # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
    save_dir=Path(""), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
    plots=True,  # 是否可视化 运行val.py传入默认True 
    callbacks=Callbacks(), 
    compute_loss=None, # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
):
```

### 3.2 Initialize/load model and set device（初始化/加载模型以及设置设备）


```python
  if training:  # called by train.py 通过train.py调用的run函数
        device, of, engine = (
            next(model.parameters()).device,
            True,
            False,
        )  # get model device, OneFlow model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly 通过val.py 调用的run函数
        device = select_device(device, batch_size=batch_size)

        # Directories  生成save_dir文件路径  run/test/expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model 加载模型 
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        
        stride, of, engine = model.stride, model.of, model.engine
        # 检测输入图片的分辨率imgsz是否能被gs整除 
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not of:
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 inference (1,3,{imgsz},{imgsz}) for non-OneFlow models")
        
        # Data
        data = check_dataset(data)  # check
```

### 3.3 Configure


```python
# 配置
model.eval() # 启动模型验证模式
cuda = device.type != "cpu"
is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
nc = 1 if single_cls else int(data["nc"])  # number of classes
# iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
iouv = flow.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel() # 示例 mAP@0.5:0.95 iou个数=10个
```

### 3.4 Dataloader
> 通过train.py调用run函数会传入一个Dataloader，而通过val.py需要加载测试数据集


```python
# Dataloader
if not training: # 加载val数据集
    if of and not single_cls:  # check --weights are trained on --data
        ncm = model.model.nc
        assert ncm == nc, (
            f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} " f"classes). Pass correct combination of" f" --weights and --data that are trained together."
        )
    model.warmup(imgsz=(1 if of else batch_size, 3, imgsz, imgsz))  # warmup
    pad = 0.0 if task in ("speed", "benchmark") else 0.5
    rect = False if task == "benchmark" else of  # square inference for benchmarks
    task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
    # 创建dataloader 这里的rect默认为True 矩形推理用于测试集 在不影响mAP的情况下可以大大提升推理速度
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]
```

### 3.5 初始化


```python
# 初始化验证的图片的数量
seen = 0
# 初始化混淆矩阵
confusion_matrix = ConfusionMatrix(nc=nc)

#  获取数据集所有类别的类名
names = dict(enumerate(model.names if hasattr(model, "names") else model.module.names))

# coco80_to_coco91_class :  converts 80-index (val2014) to 91-index (paper) 
# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
# 设置进度条模块显示信息
s = ("%20s" + "%11s" * 6) % (
    "Class",
    "Images",
    "Labels",
    "P",
    "R",
    "mAP@.5",
    "mAP@.5:.95",
)
# 初始化时间dt[t0, t1, t2] 和 p, r, f1, mp, mr, map50, map指标
dt, p, r, f1, mp, mr, map50, map = (
    [0.0, 0.0, 0.0],
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
#  初始化验证集的损失
loss = flow.zeros(3, device=device)
#  初始化json文件中的字典 统计信息 ap ap_class 
jdict, stats, ap, ap_class = [], [], [], []
callbacks.run("on_val_start")
# 初始化tqdm 进度条模块
pbar = tqdm(dataloader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
```
<details open>
<summary>示例输出 </summary>

```python
val: data=data/coco.yaml, weights=['yolov5x'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, 
    device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, 
    save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v1.0-8-g94ec5c4 Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|████████
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 157/157 [01:55<00:00,  1.36it/
                 all       5000      36335      0.743      0.627      0.685      0.503
Speed: 0.1ms pre-process, 7.5ms inference, 2.1ms NMS per image at shape (32, 3, 640, 640)  # <--- baseline speed

Evaluating pycocotools mAP... saving runs/val/exp3/yolov5x_predictions.json...

...
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505 # <--- baseline mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.677  # <--- baseline mAR
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.826
```
</details>



### 3.6 开始验证


```python
for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
""" https://github.com/Oneflow-Inc/one-yolov5/blob/bf8c66e011fcf5b8885068074ffc6b56c113a20c/utils/dataloaders.py#L735
im :  flow.from_numpy(img);
targets : labels_out 
paths: self.im_files[index] 
shapes : shapes
"""
```

#### 3.6.1 验证开始前的预处理


```python
callbacks.run("on_val_batch_start")
t1 = time_sync()
if cuda:
    im = im.to(device)
    targets = targets.to(device)
im = im.half() if half else im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
nb, _, height, width = im.shape  # batch size, channels, height, width
t2 = time_sync()
dt[0] += t2 - t1
```

#### 3.6.2 推理


```python
# Inference
out, train_out = model(im) if training else model(im, augment=augment, val=True)  # 输出为：推理结果、损失值
dt[1] += time_sync() - t2
```

#### 3.6.3 计算损失


```python
# Loss
"""
分类损失(cls_loss)：该损失用于判断模型是否能够准确地识别出图像中的对象，并将其分类到正确的类别中。

置信度损失(obj_loss)：该损失用于衡量模型预测的框（即包含对象的矩形）与真实框之间的差异。

边界框损失(box_loss)：该损失用于衡量模型预测的边界框与真实边界框之间的差异，这有助于确保模型能够准确地定位对象。
"""
if compute_loss:
    loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
```

#### 3.6.4 Run NMS


```python
# NMS
# 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
targets[:, 2:] *= flow.tensor((width, height, width, height), device=device)  # to pixels
# 对应
lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
t3 = time_sync()
"""non_max_suppression (非最大值抑制)
Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
该算法的原理：
先假设有6个矩形框，根据分类器的类别分类概率大小排序，假设从小到大属于车辆(被检测的目标)的概率分别为：A、B、C、D、E、F
（1）从最大概率 矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个指定的阀值；
（2）假设B、D与F的重叠度大于指定的阀值，则丢弃B、D，并标记第一个矩形框 F，使我们要保留的
（3）从剩下的矩形框A、C、E中，选择最大概率，假设为E，然后判断A、C与E的重叠度是否大于指定的阀值，
     假如大于就丢弃A、C，并标记E，是我们保留下来的第二个矩形框
一直重复上述过程，找到所有被保留的矩形框
Returns:
     list of detections, on (n,6) tensor per image [xyxy, conf, cls]
"""
out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
#  获取NMS时间
dt[2] += time_sync() - t3
```

#### 3.6.5 统计每张图片的真实框、预测框信息 


```python
# Metrics
for si, pred in enumerate(out):
    labels = targets[targets[:, 0] == si, 1:]
    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
    path, shape = Path(paths[si]), shapes[si][0]
    correct = flow.zeros(npr, niou, dtype=flow.bool, device=device)  # init
    seen += 1 # 图片数量 +1

    if npr == 0:# 如果预测为空，则添加空的信息到stats里
        if nl:
            stats.append((correct, *flow.zeros((2, 0), device=device), labels[:, 0]))
            if plots:
                confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
        continue
        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        # 将预测坐标映射到原图img中
        scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = flow.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            if plots:
                confusion_matrix.process_batch(predn, labelsn)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # Save/log
        if save_txt:
            save_one_txt(
                predn,
                save_conf,
                shape,
                file=save_dir / "labels" / f"{path.stem}.txt",
            )
        if save_json:
            save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
        callbacks.run("on_val_image_end", pred, predn, path, names, im[si])
```

### 3.6.6 画出前三个batch图片的gt和pred框
> gt : 真实框，Ground truth box, 是人工标注的位置，存放在标注文件中

> pred : 预测框，Prediction box， 是由目标检测模型计算输出的框


```python
# Plot images
if plots and batch_i < 3:
    plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
    plot_images(
        im,
        output_to_target(out),
        paths,
        save_dir / f"val_batch{batch_i}_pred.jpg",
        names,
    )  # pred

callbacks.run("on_val_batch_end")
```

### 3.7 计算指标
> 指标名字在代码中体现


```python
# Compute metrics
stats = [flow.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
```

### 3.8 打印日志


```python
# Print results per class
if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
for i, c in enumerate(ap_class):
    LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

# Print speeds
t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
if not training:
shape = (batch_size, 3, imgsz, imgsz)
LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)
```

### 3.9保存验证结果


```python
# Plots
if plots:
    confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    callbacks.run("on_val_end")

# Save JSON
if save_json and len(jdict):
    w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
    anno_json = str(Path(data.get("path", "../coco")) / "annotations/instances_val2017.json")  # annotations json
    pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
    with open(pred_json, "w") as f:
        json.dump(jdict, f)
    # try-catch，会有哪些error
    """
    pycocotools介绍:
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    尝试:
        使用pycocotools工具计算loss
        COCO API - http://cocodataset.org/
    失败error:
        直接打印抛出的异常
        1. 可能没有安装 pycocotools，但是网络有问题，无法实现自动下载。
        2. pycocotools包版本有问题
    """
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        check_requirements(["pycocotools"])
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, "bbox")
        if is_coco:
            eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        LOGGER.info(f"pycocotools unable to run: {e}")
```

### 3.10 返回结果


```python
# Return results
model.float()  # for training
if not training:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
maps = np.zeros(nc) + map
for i, c in enumerate(ap_class):
    maps[c] = ap[i]
return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
```

##  Reference

- [【什么是epoch、batch、batchsize、iteration？什么是真实框、预测框和锚框】](https://blog.csdn.net/qq_29960631/article/details/121945133?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-121945133-blog-104170604.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-121945133-blog-104170604.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=14)


- [【YOLOV5-5.x 源码解读】val.py](https://blog.csdn.net/qq_38253797/article/details/119577291)
