## åè¨

>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å <a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" > ææ°çå¨æã </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  > å¦æä½ æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹ä½ æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

## æ¨¡åèå (Model Ensembling)

From https://www.sciencedirect.com/topics/computer-science/ensemble-modeling:
>   Ensemble modeling is a process where multiple diverse models are created to predict an outcome, either by using many different modeling algorithms or using different training data sets. The ensemble model then aggregates the prediction of each base model and results in once final prediction for the unseen data. The motivation for using ensemble models is to reduce the generalization error of the prediction. As long as the base models are diverse and independent, the prediction error of the model decreases when the ensemble approach is used. The approach seeks the wisdom of crowds in making a prediction. Even though the ensemble model has multiple base models within the model, it acts and performs as a single model.

ð è¿ä¸ªæç¨ç¨æ¥è§£éå¨YOLOv5æ¨¡åçæµè¯åæ¨çä¸­å¦ä½ä½¿ç¨æ¨¡åèå (Model Ensembling)æé«mAPåRecall ð 

### ðå¼å§ä¹å
åéå·¥ç¨å¹¶å¨ [Python>3.7.0](https://www.python.org/) çç¯å¢ä¸­å®è£ [requiresments.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow è¯·éæ© [nightly çæ¬æè >0.9 çæ¬](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) ã[æ¨¡å](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)å[æ°æ®](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)å¯ä»¥ä»æºç ä¸­èªå¨ä¸è½½ã

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5.git
cd one-yolov5
pip install -r requirements.txt  # install
```


### ðæ®éæµè¯

å¨å°è¯`TTA`ä¹åï¼æä»¬å¸æå»ºç«ä¸ä¸ªåºåè½å¤è¿è¡æ¯è¾ãè¯¥å½ä»¤å¨COCO val2017ä¸ä»¥640åç´ çå¾åå¤§å°æµè¯YOLOv5xã `yolov5x` æ¯å¯ç¨çæå¤§å¹¶ä¸æç²¾ç¡®çæ¨¡åãå¶å®å¯ç¨çæ¨¡åæ¯ `yolov5s`, `yolov5m`  å `yolov5l`ç­  æè èªå·±ä»æ°æ®éè®­ç»åºçæ¨¡å `./weights/best`ãæå³ææå¯ç¨æ¨¡åçè¯¦ç»ä¿¡æ¯ï¼è¯·åéæä»¬ç [READEME table](https://github.com/Oneflow-Inc/one-yolov5#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A3%80%E6%9F%A5%E7%82%B9)

```python
$ python val.py --weights ./yolov5x --data coco.yaml --img 640 --half
```

ð¢ è¾åº:
```shell
val: data=data/coco.yaml, weights=['./yolov5x'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 ð v1.0-8-g94ec5c4 Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|ââââââââ
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|ââââââââââ| 157/157 [01:55<00:00,  1.36it/
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

### ð èåæµè¯

éè¿å¨ä»»ä½ç°æç val.pyædetect.pyå½ä»¤ä¸­ç `--weights` åæ°åæ·»å é¢å¤çæ¨¡åï¼å¯ä»¥å¨æµè¯åæ¨çæ¶å°å¤ä¸ªé¢è®­ç»æ¨¡åèåå¨ä¸èµ·ã

ð¢ å° `yolov5x`,`yolov5l6` ä¸¤ä¸ªæ¨¡åçèåæµè¯çæä»¤å¦ä¸ï¼
```
python val.py --weights ./yolov5x ./yolov5l6  --data data/coco.yaml --img 640 --half
```

```
val: data=data/coco.yaml, weights=['./yolov5x', './yolov5l6'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 ð v1.0-29-g8ed33f3 Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
Fusing layers... 
Model summary: 346 layers, 76726332 parameters, 653820 gradients
Ensemble created with ['./yolov5x', './yolov5l6']

val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|ââââââââ
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|ââââââââââ| 157/157 [03:14<00:00,  1.24s/i
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

ð¢ å£°æ:ä¸è¿°ä¸¤æ¬¡æµè¯çmAPï¼mARç»æå¦ä¸ï¼

|          | mAP   | mAR   |
|----------|-------|-------|
| baseline | 0.505 | 0.677 |
| ensemble | 0.515 | 0.690 |

### ðèåæ¨ç

éå é¢å¤çæ¨¡åå¨ `--weights` éé¡¹åèªå¨å¯ç¨èåæ¨çï¼

```
python detect.py --weights ./yolov5x ./yolov5l6 --img 640 --source  data/images
```
Output:
```
detect: weights=['./yolov5x', './yolov5l6'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 ð v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221018+cu112 
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

![image](https://user-images.githubusercontent.com/109639975/202892132-f6d59244-1f5c-4175-927d-813ab2907ab1.png)


### åèæç« 

- https://github.com/ultralytics/yolov5/issues/318
