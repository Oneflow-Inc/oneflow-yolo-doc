## 模型融合 (Model Ensembling)

From https://www.sciencedirect.com/topics/computer-science/ensemble-modeling:
>   Ensemble modeling is a process where multiple diverse models are created to predict an outcome, either by using many different modeling algorithms or using different training data sets. The ensemble model then aggregates the prediction of each base model and results in once final prediction for the unseen data. The motivation for using ensemble models is to reduce the generalization error of the prediction. As long as the base models are diverse and independent, the prediction error of the model decreases when the ensemble approach is used. The approach seeks the wisdom of crowds in making a prediction. Even though the ensemble model has multiple base models within the model, it acts and performs as a single model.

📚 这个教程用来解释在YOLOv5模型的测试和推理中如何使用模型融合 (Model Ensembling)提高mAP和Recall 🚀

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
$ python val.py --weights ./yolov5x --data coco.yaml --img 640 --half
```

📢 输出:
```shell
val: data=data/coco.yaml, weights=['./yolov5x'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
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

### 📌 融合测试

通过在任何现有的 val.py或detect.py命令中的 `--weights` 参数后添加额外的模型，可以在测试和推理时将多个预训练模型融合合在一起。

📢 将 `yolov5x`,`yolov5l6` 两个模型的融合测试的指令如下：
```
python val.py --weights ./yolov5x ./yolov5l6  --data data/coco.yaml --img 640 --half
```

```
val: data=data/coco.yaml, weights=['./yolov5x', './yolov5l6'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v1.0-29-g8ed33f3 Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
Fusing layers... 
Model summary: 346 layers, 76726332 parameters, 653820 gradients
Ensemble created with ['./yolov5x', './yolov5l6']

val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|████████
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 157/157 [03:14<00:00,  1.24s/i
                 all       5000      36335       0.73      0.644      0.693      0.513
Speed: 0.1ms pre-process, 23.7ms inference, 2.3ms NMS per image at shape (32, 3, 640, 640) # <--- ensemble speed

Evaluating pycocotools mAP... saving runs/val/exp21/yolov5x_predictions.json...

...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515  # <--- ensemble mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.697
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.690 # <--- ensemble mAR
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.842
```

📢 声明:上述两次测试的mAP，mAR结果如下：

|          | mAP   | mAR   |
|----------|-------|-------|
| baseline | 0.505 | 0.677 |
| ensemble | 0.515 | 0.690 |

### 📌融合推理

附加额外的模型在 `--weights` 选项后自动启用融合推理：

```
python detect.py --weights ./yolov5x ./yolov5l6 --img 640 --source  data/images
```
Output:
```
detect: weights=['./yolov5x', './yolov5l6'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 🚀 v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
Fusing layers... 
Model summary: 346 layers, 76726332 parameters, 653820 gradients
Ensemble created with ['./yolov5x', './yolov5l6']

detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 1/2 /home/fengwen/one-yolov5/data/images/bus.jpg: 640x512 4 persons, 1 bus, 1 handbag, 1 tie, Done. (0.028s)
detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 2/2 /home/fengwen/one-yolov5/data/images/zidane.jpg: 384x640 3 persons, 2 ties, Done. (0.023s)
0.6ms pre-process, 25.6ms inference, 2.4ms NMS per image at shape (1, 3, 640, 640)
```
<img src="/tutorials/03_chapter/model_ensembling_imgs/zidane.jpg">

### 参考文章

- https://github.com/ultralytics/yolov5/issues/318
