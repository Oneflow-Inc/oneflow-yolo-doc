
## 测试时数据增强 🚀

> 📚 这个教程用来解释在YOLOv5模型的测试和推理中如何使用 Test Time Augmentation (TTA) 提高mAP和Recall 🚀。

### 📌开始之前

克隆工程并在 [Python>3.7.0](https://www.python.org/) 的环境中安装 [requiresments.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow 请选择 [nightly 版本或者 >0.9 版本](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) 。[模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)和[数据](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)可以从源码中自动下载。

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5.git
cd one-yolov5
pip install -r requirements.txt  # install
```

### 📌普通测试

在尝试`TTA`之前，我们希望建立一个基准能够进行比较。该命令在COCO val2017上以640像素的图像大小测试YOLOv5x。 `yolov5x` 是可用的最大并且最精确的模型。其它可用的是 `yolov5s`, `yolov5m`  和 `yolov5l`  或者 自己从数据集训练出的模型。`./weights/best`。有关所有可用模型的详细信息，请参阅我们的 [READEME table](https://github.com/Oneflow-Inc/one-yolov5#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A3%80%E6%9F%A5%E7%82%B9)

```python
$ python val.py --weights yolov5x --data coco.yaml --img 640 --half
```

📢 输出:
```shell
val: data=data/coco.yaml, weights=['yolov5x'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
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


### 📌TTA测试
在val.py 后附加 --augment 选项启用TTA。(`将图像大小增加约30%左右可以获得更好的结果哦`🚀)。

❗请注意: 启用TTA的推断通常需要正常推断时间的2-3倍，因为图像左右翻转并以3种不同分辨率处理，输出在NMS之前合并。

速度下降的部分原因是图像尺寸较大（832 vs 640），当然也有部分原因是 TTA 操作造成的。

```python
$ python val.py --weights yolov5x --data coco.yaml --img 832 --augment --half
```

输出:
```python
(python3.8) fengwen@oneflow-25:~/one-yolov5$ python val.py --weights yolov5x --data data/coco.yaml  --img 832 --augment --half
loaded library: /lib/x86_64-linux-gnu/libibverbs.so.1
val: data=data/coco.yaml, weights=['yolov5x'], batch_size=32, imgsz=832, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=True, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221021+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|██████████| 
            Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 157/157 [04:39<00:00,  1.78s/it]   
              all       5000      36335      0.743      0.645        0.7      0.518
Speed: 0.1ms pre-process, 40.6ms inference, 2.2ms NMS per image at shape (32, 3, 832, 832)

Evaluating pycocotools mAP... saving runs/val/exp/yolov5x_predictions.json...
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.519 # <--- TTA mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.704
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.662
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.698 # <--- TTA mAR
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.745
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.837
```

📢 声明:上述两次测试的mAP，mAR结果如下：
|          | mAP   | mAR   |
|----------|-------|-------|
| baseline | 0.505 | 0.677 |
| TTA      | 0.519 | 0.698 |

### 📌TTA推理

在 detect.py 中使用 TTA 的操作与 val.py 中使用TTA相同：只需将其附加 --augment 到任何现有检测任务中。
detect.py 指令「案例🌰」:
```python
$ python detect.py --weights yolov5s --img 832 --source data/images --augment
```
输出:
```
loaded library: /lib/x86_64-linux-gnu/libibverbs.so.1
detect: weights=['yolov5x'], source=data/images/, data=data/coco128.yaml, imgsz=[832, 832], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=True, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 🚀 v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221021+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 1/2 /home/fengwen/one-yolov5/data/images/bus.jpg: 832x640 4 persons, 1 bicycle, 1 bus, Done. (0.057s)
detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 2/2 /home/fengwen/one-yolov5/data/images/zidane.jpg: 480x832 3 persons, 2 ties, Done. (0.041s)
0.5ms pre-process, 48.6ms inference, 2.1ms NMS per image at shape (1, 3, 832, 832)
```
<img src="TTA_imgs/zidane.jpg">


# OneFlow Hub TTA
TTA自动集成到所有YOLOv5 OneFlow Hub模型中，并可在推理时通过传递 augment=True 参数进行开启。
```python
import oneflow as flow

# 模型
model = flow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# 图像
img = 'https://raw.githubusercontent.com/Oneflow-Inc/one-yolov5/main/data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list


# 推理
results = model(img)

# 结果
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```


# 自定义
我们可以自定义TTA操作在 YOLOv5 **forward_augment()** 方法中, 应用的TTA操作细节具体可见：

https://github.com/Oneflow-Inc/one-yolov5/blob/bbdf286ad1b1d3fd2c82cecdfa4487db423d9cfe/models/yolo.py#L141-L153


### 参考文章

- https://github.com/ultralytics/yolov5/issues/303
