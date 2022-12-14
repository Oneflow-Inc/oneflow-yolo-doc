## åè¨

>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å <a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" > ææ°çå¨æã </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  > å¦ææ¨æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹æ¨æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

æ¬æä¸»è¦ä»ç» one-yolov5 ä½¿ç¨çæ°æ®éçæ ¼å¼ä»¥åå¦ä½å¶ä½ä¸ä¸ªå¯ä»¥è·å¾æ´å¥½è®­ç»ææçæ°æ®éãæ¬èæç¨å¥½çæ°æ®éæ åé¨åç¿»è¯äº ultralytics/yolov5 wiki ä¸­[å¯¹æ°æ®éç¸å³çæè¿°](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)ã


# æ°æ®éç»æè§£è¯»
## 1.åå»ºdataset.yaml

COCO128æ¯å®æ¹ç»çä¸ä¸ªå°çæ°æ®é ç±[COCO](https://cocodataset.org/#home) æ°æ®éå 128 å¼ å¾çç»æã
è¿128å¹å¾åç¨äºè®­ç»åéªè¯ï¼å¤æ­ yolov5 èæ¬æ¯å¦è½å¤è¿æ­£å¸¸è¿è¡ã
[æ°æ®ééç½®æä»¶ coco128.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/master/data/coco128.yaml) å®ä¹äºå¦ä¸çéç½®éé¡¹ï¼ 

```coco128.yaml
# YOLOv5 ð by Ultralytics, GPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: python train.py --data coco128.yaml
# parent
# âââ one-yolov5
# âââ datasets
#     âââ coco128  â downloads here (7 MB)

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]

# è®­ç»åéªè¯å¾åçè·¯å¾ç¸å
train: ../coco128/images/train2017/ 
val: ../coco128/images/train2017/

# number of classes
nc: 80 # ç±»å«æ°

# class names ç±»ååè¡¨
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']


# Download script/URL (optional) ç¨äºèªå¨ä¸è½½çå¯éä¸è½½å½ä»¤/URL ã 
download: https://ultralytics.com/assets/coco128.zip
```
 
 æ³¨æï¼å¦ææ¯èªå®ä¹æ°æ®éçè¯æèªå·±éæ±ä¿®æ¹è¿ä¸ªyamlæä»¶ãä¸»è¦ä¿®æ¹ä»¥ä¸ä¸¤ç¹ã
 1. ä¿®æ¹è®­ç»åéªè¯å¾åçè·¯å¾ä¸ºèªå®ä¹æ°æ®éè·¯å¾
 2. ä¿®æ¹ç±»å«æ°åç±»ååè¡¨

 åå±ç¤ºä¸ä¸ `coco.yaml` çæ°æ®éè·¯å¾éç½®ï¼è¿éçè®­ç»åéªè¯å¾åçè·¯å¾å°±æ¯ç´æ¥ç¨txtè¡¨ç¤ºï¼

![å¾ç](https://user-images.githubusercontent.com/35585791/200171483-449db3a4-a813-4169-9dcc-32386ea477f7.png)

 
## 2.åå»º Labels
ä½¿ç¨å·¥å·ä¾å¦ [CVAT](https://github.com/opencv/cvat) , [makesense.ai](https://www.makesense.ai/), [Labelbox](https://labelbox.com/) ï¼LabelImg(å¨æ¬ç« å¦ä½å¶ä½æ°æ®éä¸­ä»ç»LabelImgå·¥å·ä½¿ç¨) ç­ï¼å¨ä½ èªå·±çæ°æ®éæä¾çå¾çä¸åç®æ æ¡çæ æ³¨ï¼å°æ æ³¨ä¿¡æ¯å¯¼åºä¸ºä¸ä¸ªtxtåç¼ç»å°¾çæä»¶ãï¼å¦æå¾åä¸­æ²¡æç®æ ï¼åä¸éè¦*.txtæä»¶ï¼ã

*.txtæä»¶è§èå¦ä¸æç¤º:
- æ¯ä¸è¡ ä¸ä¸ªç®æ ã
- æ¯ä¸è¡æ¯ class x_center y_center width height æ ¼å¼ã
- æ¡åæ å¿é¡»éç¨æ ååxywhæ ¼å¼ï¼ä»0å°1ï¼ãå¦ææ¡ä»¥åç´ ä¸ºåä½ï¼åå°x_centeråwidthé¤ä»¥å¾åå®½åº¦ï¼å°y_centreåheighté¤ä»¥å¾åé«åº¦ã
- ç±»å·ä¸ºé¶ç´¢å¼çç¼å·ï¼ä»0å¼å§è®¡æ°ï¼ã


<p align="center">
  <img src="https://user-images.githubusercontent.com/35585791/200169794-1fa5209c-44db-4a25-9456-29de2d5674f1.png">
  
  **è¿éåè®¾ä»¥ COCO æ°æ®éçç®æ ç±»å«çº¦å®æ¥æ æ³¨**
</p>

ä¸ä¸è¿°å¾åç¸å¯¹åºçæ ç­¾æä»¶åå«2ä¸ªäººï¼class 0ï¼å ä¸ä¸ªé¢å¸¦ï¼class 27ï¼ï¼


![imgs](https://user-images.githubusercontent.com/35585791/200169914-e8ae3413-e4d5-4a8c-bd1f-12ef7200b72c.png)

## 3.COCO128 æ°æ®éç®å½ç»æç»ç»

å¨æ¬ä¾ä¸­ï¼æä»¬ç **coco128** æ¯ä½äº **yolov5** ç®å½éè¿ãyolov5 éè¿å°æ¯ä¸ªå¾åè·¯å¾ **xx/images/xx.jpg** æ¿æ¢ä¸º **xx/labels/xx.txt** æ¥èªå¨å®ä½æ¯ä¸ªå¾åçæ ç­¾ãä¾å¦ï¼
```Python
dataset/images/im0.jpg  # image
dataset/labels/im0.txt  # label
```
![coco å coco128 æ°æ®éç»ç»ç»æ](https://user-images.githubusercontent.com/35585791/200170263-e7cfc4b9-1271-4c38-8ebd-653556e0cf60.png)


# å¶ä½æ°æ®é

## æ°æ®éæ æ³¨å·¥å·
è¿éä¸»è¦ä»ç» LabelImg: æ¯ä¸ç§ç©å½¢æ æ³¨å·¥å·ï¼å¸¸ç¨äºç®æ è¯å«åç®æ æ£æµ,å¯ç´æ¥çæ yolov5 è¯»åçtxtæ ç­¾æ ¼å¼ï¼ä½å¶åªè½è¿è¡ç©å½¢æ¡æ æ³¨ã(å½ç¶ä¹å¯ä»¥éç¨å¶å®çå·¥å·è¿è¡æ æ³¨å¹¶ä¸ç½ä¸é½æå¤§éå³äºæ æ³¨å·¥å·çæç¨ã)

é¦ålabelimgçå®è£ååç®åï¼ç´æ¥ä½¿ç¨cmdä¸­çpipè¿è¡å®è£ï¼å¨cmdä¸­è¾å¥å½ä»¤è¡ï¼
```python3
pip install labelimg
```
å®è£åç´æ¥è¾å¥å½ä»¤ï¼
```
labelimg
```
å³å¯æå¼è¿è¡ã

ç¹å»Open Diréæ©æ°æ®éæä»¶å¤¹ï¼åç¹å»Create RectBoxè¿è¡æ æ³¨ã

![å¾ç](https://user-images.githubusercontent.com/35585791/200170723-559ceea0-5473-4c97-99e9-a5c5f4a16167.png)


å½ä½ ç»å¶æ¡ç»æå°±ä¼å¼¹åºæ ç­¾éæ©æ¡ï¼ç¶åæ æ³¨ç±»å«ãè¿ä¸ªç±»å«ç¼è¾æ´æ¹å¨Labelimgæä»¶éï¼éé¢æclasses.txtææ¡£ï¼æå¼æå¨æ´æ¹ç±»å«å³å¯ãï¼å½åºç°æ°ç±»å«æ¶ä¹å¯å¨æ ç­¾éæ©æ¡éè¾å¥ç¹OKå°±èªå¨æ·»å ç±»å«äºï¼

æ æ³¨å¥½åéæ© yolo æ ¼å¼ï¼ç¹å» Save ä¿å­ãæ æ³¨ç»æä¿å­å¨`å¾çå.txt`æä»¶ä¸­ï¼txtæä»¶åå¾çåç§°ä¸è´ï¼åå®¹å¦ä¸ï¼

<p align="center">
  <img src="https://user-images.githubusercontent.com/35585791/200170815-f4a6b66e-b7b8-486b-8641-099020e60c69.png">
</p>


## ä¸ä¸ªå¥½çæ°æ®éæ åï¼

- æ¯ä¸ªç±»çå¾åã >= 1500 å¼ å¾çã
- æ¯ä¸ªç±»çå®ä¾ãâ¥ å»ºè®®æ¯ä¸ªç±»10000ä¸ªå®ä¾ï¼æ è®°å¯¹è±¡ï¼
- å¾çå½¢è±¡å¤æ ·ãå¿é¡»ä»£è¡¨å·²é¨ç½²çç¯å¢ãå¯¹äºç°å®ä¸ççä½¿ç¨æ¡ä¾ï¼æä»¬æ¨èæ¥èªä¸å¤©ä¸­ä¸åæ¶é´ãä¸åå­£èãä¸åå¤©æ°ãä¸åç§æãä¸åè§åº¦ãä¸åæ¥æºï¼å¨çº¿ééãæ¬å°ééãä¸åæåæºï¼ç­çå¾åã
- æ ç­¾ä¸è´æ§ãå¿é¡»æ è®°ææå¾åä¸­ææç±»çææå®ä¾ãé¨åæ è®°å°ä¸èµ·ä½ç¨ã
- æ ç­¾åç¡®æ§ã
- æ ç­¾å¿é¡»ç´§å¯å°åå´æ¯ä¸ªå¯¹è±¡ãå¯¹è±¡ä¸å¶è¾¹çæ¡ä¹é´ä¸åºå­å¨ä»»ä½ç©ºé´ãä»»ä½å¯¹è±¡é½ä¸åºç¼ºå°æ ç­¾ã
- æ ç­¾éªè¯ãæ¥çtrain_batch*.jpg å¨ è®­ç»å¼å§éªè¯æ ç­¾æ¯å¦æ­£ç¡®ï¼å³åè§ mosaic ï¼å¨ yolov5 çè®­ç»æ¥å¿ runs/train/exp* æä»¶å¤¹éé¢å¯ä»¥çå°ï¼ã
- èæ¯å¾åãèæ¯å¾åæ¯æ²¡ææ·»å å°æ°æ®éä»¥åå° False Positivesï¼FPï¼çå¯¹è±¡çå¾åãæä»¬å»ºè®®ä½¿ç¨å¤§çº¦0-10%çèæ¯å¾åæ¥å¸®å©åå°FPsï¼COCOæ1000ä¸ªèæ¯å¾åä¾åèï¼å æ»æ°ç1%ï¼ãèæ¯å¾åä¸éè¦æ ç­¾ã

ä¸å¾å±ç¤ºäºå¤ç§æ°æ®éçæ ç­¾ç¹ç¹ï¼

<p align="center">
  <a href= "https://arxiv.org/abs/1405.0312">
  <img src="https://user-images.githubusercontent.com/26833433/109398377-82b0ac00-78f1-11eb-9c76-cc7820669d0d.png">
  </a>  
</p>

å¶ä¸­ï¼

- Instances per category è¡¨ç¤ºæ¯ä¸ªç±»å«çå®ä¾æ°
- Categories per image è¡¨ç¤ºæ¯å¹å¾åçç±»å«
- (a) Instances per image è¡¨ç¤ºæ¯å¹å¾åçå®ä¾æ°
- (b) Number of categories vs. number of instances è¡¨ç¤ºç±»å«æ°ç® vs å®ä¾æ°ç® ï¼æä»¬å¯ä»¥çå° COCO æ°æ®éçç±»å«åå®ä¾çæ°ç®è¾¾å°äºä¸ä¸ªè¾å¥½çå¹³è¡¡ï¼
- (c) Instance size è¡¨ç¤ºå®ä¾ä¸ªæ°
- (d) Number of categories è¡¨ç¤ºç±»å«æ°
- (e) Percent of image size è¡¨ç¤ºå¾åå¤§å°ç¾åæ¯

## åèæç« 
- https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
- https://docs.ultralytics.com/tutorials/train-custom-datasets/#weights-biases-logging-new

