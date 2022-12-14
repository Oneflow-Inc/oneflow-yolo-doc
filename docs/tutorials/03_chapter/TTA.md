## åè¨

>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å <a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" > ææ°çå¨æã </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  > å¦æä½ æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹ä½ æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

## æµè¯æ¶æ°æ®å¢å¼º ð

> ð è¿ä¸ªæç¨ç¨æ¥è§£éå¨YOLOv5æ¨¡åçæµè¯åæ¨çä¸­å¦ä½ä½¿ç¨ Test Time Augmentation (TTA) æé«mAPåRecall ðã

### ðå¼å§ä¹å

åéå·¥ç¨å¹¶å¨ [Python>3.7.0](https://www.python.org/) çç¯å¢ä¸­å®è£ [requiresments.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow è¯·éæ© [nightly çæ¬æè >0.9 çæ¬](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) ã[æ¨¡å](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)å[æ°æ®](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)å¯ä»¥ä»æºç ä¸­èªå¨ä¸è½½ã

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5.git
cd one-yolov5
pip install -r requirements.txt  # install
```

### ðæ®éæµè¯

å¨å°è¯`TTA`ä¹åï¼æä»¬å¸æå»ºç«ä¸ä¸ªåºåè½å¤è¿è¡æ¯è¾ãè¯¥å½ä»¤å¨COCO val2017ä¸ä»¥640åç´ çå¾åå¤§å°æµè¯YOLOv5xã `yolov5x` æ¯å¯ç¨çæå¤§å¹¶ä¸æç²¾ç¡®çæ¨¡åãå¶å®å¯ç¨çæ¯ `yolov5s`, `yolov5m`  å `yolov5l`  æè èªå·±ä»æ°æ®éè®­ç»åºçæ¨¡åã`./weights/best`ãæå³ææå¯ç¨æ¨¡åçè¯¦ç»ä¿¡æ¯ï¼è¯·åéæä»¬ç [READEME table](https://github.com/Oneflow-Inc/one-yolov5#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A3%80%E6%9F%A5%E7%82%B9)

```python
$ python val.py --weights yolov5x --data coco.yaml --img 640 --half
```

ð¢ è¾åº:
```shell
val: data=data/coco.yaml, weights=['yolov5x'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
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


### ðTTAæµè¯
å¨val.py åéå  --augment éé¡¹å¯ç¨TTAã(`å°å¾åå¤§å°å¢å çº¦30%å·¦å³å¯ä»¥è·å¾æ´å¥½çç»æå¦`ð)ã

âè¯·æ³¨æ: å¯ç¨TTAçæ¨æ­éå¸¸éè¦æ­£å¸¸æ¨æ­æ¶é´ç2-3åï¼å ä¸ºå¾åå·¦å³ç¿»è½¬å¹¶ä»¥3ç§ä¸ååè¾¨çå¤çï¼è¾åºå¨NMSä¹ååå¹¶ã

éåº¦ä¸éçé¨ååå æ¯å¾åå°ºå¯¸è¾å¤§ï¼832 vs 640ï¼ï¼å½ç¶ä¹æé¨ååå æ¯ TTA æä½é æçã

```python
$ python val.py --weights yolov5x --data coco.yaml --img 832 --augment --half
```

è¾åº:
```python
(python3.8) fengwen@oneflow-25:~/one-yolov5$ python val.py --weights yolov5x --data data/coco.yaml  --img 832 --augment --half
loaded library: /lib/x86_64-linux-gnu/libibverbs.so.1
val: data=data/coco.yaml, weights=['yolov5x'], batch_size=32, imgsz=832, conf_thres=0.001, iou_thres=0.6, task=val, device=, workers=8, single_cls=False, augment=True, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 ð v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221021+cu112 
Fusing layers... 
Model summary: 322 layers, 86705005 parameters, 571965 gradients
val: Scanning '/data/dataset/fengwen/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100%|ââââââââââ| 
            Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|ââââââââââ| 157/157 [04:39<00:00,  1.78s/it]   
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

ð¢ å£°æ:ä¸è¿°ä¸¤æ¬¡æµè¯çmAPï¼mARç»æå¦ä¸ï¼
|          | mAP   | mAR   |
|----------|-------|-------|
| baseline | 0.505 | 0.677 |
| TTA      | 0.519 | 0.698 |

### ðTTAæ¨ç

å¨ detect.py ä¸­ä½¿ç¨ TTA çæä½ä¸ val.py ä¸­ä½¿ç¨TTAç¸åï¼åªéå°å¶éå  --augment å°ä»»ä½ç°ææ£æµä»»å¡ä¸­ã
detect.py æä»¤ãæ¡ä¾ð°ã:
```python
$ python detect.py --weights yolov5s --img 832 --source data/images --augment
```
è¾åº:
```
loaded library: /lib/x86_64-linux-gnu/libibverbs.so.1
detect: weights=['yolov5x'], source=data/images/, data=data/coco128.yaml, imgsz=[832, 832], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=True, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 ð v1.0-31-g6b1387c Python-3.8.13 oneflow-0.8.1.dev20221021+cu112 
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

![image](https://user-images.githubusercontent.com/109639975/202892096-8d3f246c-97af-478a-b03a-cf751dc6a544.png)


# OneFlow Hub TTA
TTAèªå¨éæå°ææYOLOv5 OneFlow Hubæ¨¡åä¸­ï¼å¹¶å¯å¨æ¨çæ¶éè¿ä¼ é augment=True åæ°è¿è¡å¼å¯ã
```python
import oneflow as flow

# æ¨¡å
model = flow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# å¾å
img = 'https://raw.githubusercontent.com/Oneflow-Inc/one-yolov5/main/data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list


# æ¨ç
results = model(img)

# ç»æ
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```


# èªå®ä¹
æä»¬å¯ä»¥èªå®ä¹TTAæä½å¨ YOLOv5 **forward_augment()** æ¹æ³ä¸­, åºç¨çTTAæä½ç»èå·ä½å¯è§ï¼

https://github.com/Oneflow-Inc/one-yolov5/blob/bbdf286ad1b1d3fd2c82cecdfa4487db423d9cfe/models/yolo.py#L141-L153


### åèæç« 

- https://github.com/ultralytics/yolov5/issues/303
