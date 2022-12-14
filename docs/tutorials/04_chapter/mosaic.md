>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >ææ°çå¨æã</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >å¦ææ¨æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹æ¨æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

## å¼è¨
$YOLOv5$ å¨è®­ç»æ¨¡åçæ¶åä½¿ç¨ðå¾åç©ºé´åè²å½©ç©ºé´çæ°æ®å¢å¼º(*å¨éªè¯æ¨¡åçæ¶åæ²¡æä½¿ç¨*)ï¼éè¿è®­ç»æ¶éç¨æ°æ®å¢å¼º ä»èä½¿å¾æ¯æ¬¡å è½½é½æ¯æ°çåå¯ä¸çå¾åï¼*å³åå§å¾å+3ä¸ªéæºå¾å*ï¼å¦ä¸å¾æç¤ºã

  <img src="https://user-images.githubusercontent.com/26833433/120995721-f3cfed00-c785-11eb-8ee2-b6ef2fa205e8.jpg" >

<u>å¾4.1</u> æ°æ®å¢å¼ºãç»è¿æ°æ®å¢å¼ºåå¾åä¸ä¼ä»¥ç¸åçæ¹å¼åç°ä¸¤æ¬¡ã<br> å¾çæ¥æºï¼https://docs.ultralytics.com/FAQ/augmentation/ </br>




## è¶åæ°æä»¶
æ°æ®å¢å¼ºé»è®¤ä½¿ç¨éç½®è¶åæ°æä»¶hyp.scratch.yamlï¼
ä¸é¢ä»¥[hyp.scratch-low.yaml](https://github.com/Oneflow-Inc/one-yolov5/blob/main/data/hyps/hyp.scratch-low.yaml#L22-L34)æä»¶é¨ååæ°ä¸ºä¾ï¼å·ä½åæ°è§£æå¯è§éä»¶ è¡¨2.1

```
$ python train.py --hyp hyp.scratch-low.yaml
```


### Mosaic

<img src="https://user-images.githubusercontent.com/31005897/159109235-c7aad8f2-1d4f-41f9-8d5f-b2fde6f2885e.png#pic_center" >

<u>å¾4.2</u> Mosaicæ°æ®å¢å¼ºãæ4å¼ å¾çï¼éè¿éæºç¼©æ¾ãéæºè£åãéæºæå¸çæ¹å¼è¿è¡æ¼æ¥ã



### Copy paste



  <img src="https://user-images.githubusercontent.com/31005897/159116277-91b45033-6bec-4f82-afc4-41138866628e.png#pic_center" >
  <u>å¾4.3</u> åå²å¡«è¡¥ 

### Random affine
(Rotation, Scale, Translation and Shear)ï¼æè½¬ãç¼©æ¾ãå¹³ç§»ååªåï¼


<img src="https://user-images.githubusercontent.com/31005897/159109326-45cd5acb-14fa-43e7-9235-0f21b0021c7d.png#pic_center" >

<u>å¾4.4</u>æè½¬ãç¼©æ¾ãå¹³ç§»ååªå å¾å


### MixUp 
<img src="https://user-images.githubusercontent.com/31005897/159109361-3b24333b-f481-478b-ae00-df7838f0b5cd.png#pic_center" >

<u>å¾4.5</u> å¾åèå 




### Albumentations 

YOLOv5 ð éæäºAlbumentations(ä¸ä¸ªæµè¡çå¼æºå¾åå¢å¼ºå)ã

å¯ä»¥éè¿èªå®ä¹æ°æ®éæ´å¥½å°è®­ç»ï¼ä¸çä¸ææ£çè§è§AIæ¨¡åð!

  <img src="https://user-images.githubusercontent.com/26833433/124400879-ff331b80-dd25-11eb-9b67-fe85ac4ca104.jpg" >
  <u>å¾4.6</u> Albumentations

### Augment HSV
(Hue, Saturation, Value) 
è²è°ãé¥±ååº¦ãæååº¦
<img src="https://user-images.githubusercontent.com/31005897/159109407-83d100ba-1aba-4f4b-aa03-4f048f815981.png#pic_center" >

<u>å¾4.5</u> è²è°ãé¥±ååº¦ãæååº¦



### Random horizontal flip

(éæºæ°´å¹³æç¿»è½¬)



  <img src="https://user-images.githubusercontent.com/31005897/159109429-0d44619a-a76a-49eb-bfc0-6709860c043e.png#pic_center" >
 <u>å¾4.6</u> éæºæ°´å¹³æç¿»è½¬ 


## Mosaic æ°æ®å¢å¼ºç®æ³
Mosaic æ°æ®å¢å¼ºç®æ³å°å¤å¼ å¾çæç§ä¸å®æ¯ä¾ç»åæä¸å¼ å¾çï¼ä½¿æ¨¡åå¨æ´å°çèå´åè¯å«ç®æ ãMosaic æ°æ®å¢å¼ºç®æ³åè CutMixæ°æ®å¢å¼ºç®æ³ãCutMixæ°æ®å¢å¼ºç®æ³ä½¿ç¨ä¸¤å¼ å¾çè¿è¡æ¼æ¥ï¼è Mosaic æ°æ®å¢å¼ºç®æ³ä¸è¬ä½¿ç¨åå¼ è¿è¡æ¼æ¥ï¼å¦ä¸å¾æç¤ºã
<img src ="mosaic_imgs/mosaic.png">

### Mosaicæ¹æ³æ­¥éª¤
1. éæºéåå¾çæ¼æ¥åºåç¹åæ  $(x_cï¼y_c)$ï¼å¦éæºéååå¼ å¾çã
2. åå¼ å¾çæ ¹æ®åºåç¹ï¼åå«ç»è¿ å°ºå¯¸è°æ´ å æ¯ä¾ç¼©æ¾ åï¼æ¾ç½®å¨æå®å°ºå¯¸çå¤§å¾çå·¦ä¸ï¼å³ä¸ï¼å·¦ä¸ï¼å³ä¸ä½ç½®ã
3. æ ¹æ®æ¯å¼ å¾ççå°ºå¯¸åæ¢æ¹å¼ï¼å°æ å°å³ç³»å¯¹åºå°å¾çæ ç­¾ä¸ã
4. ä¾æ®æå®çæ¨ªçºµåæ ï¼å¯¹å¤§å¾è¿è¡æ¼æ¥ãå¤çè¶è¿è¾¹ççæ£æµæ¡åæ ã

### Mosaicæ¹æ³ä¼ç¹

1. å¢å æ°æ®å¤æ ·æ§ï¼éæºéååå¼ å¾åè¿è¡ç»åï¼ç»åå¾å°å¾åä¸ªæ°æ¯åå¾ä¸ªæ°è¦å¤ã
2. å¢å¼ºæ¨¡åé²æ£æ§ï¼æ··ååå¼ å·æä¸åè¯­ä¹ä¿¡æ¯çå¾çï¼å¯ä»¥è®©æ¨¡åæ£æµè¶åºå¸¸è§è¯­å¢çç®æ ã
3. å å¼ºæ¹å½ä¸åå±ï¼$Batch \ Normalization$ï¼çææãå½æ¨¡åè®¾ç½® $BN$ æä½åï¼è®­ç»æ¶ä¼å°½å¯è½å¢å¤§æ¹æ ·æ¬æ»éï¼$BatchSize$ï¼ï¼å ä¸º $BN$ åçä¸ºè®¡ç®æ¯ä¸ä¸ªç¹å¾å±çåå¼åæ¹å·®ï¼å¦ææ¹æ ·æ¬æ»éè¶å¤§ï¼é£ä¹ $BN$ è®¡ç®çåå¼åæ¹å·®å°±è¶æ¥è¿äºæ´ä¸ªæ°æ®éçåå¼åæ¹å·®ï¼ææè¶å¥½ã

4. Mosaic æ°æ®å¢å¼ºç®æ³æå©äºæåå°ç®æ æ£æµæ§è½ãMosaic æ°æ®å¢å¼ºå¾åç±åå¼ åå§å¾åæ¼æ¥èæï¼è¿æ ·æ¯å¼ å¾åä¼ææ´å¤§æ¦çåå«å°ç®æ ã

### Mosaicæºç è§£è¯»


æè·¯æ¦æ¬ï¼å°ä¸å¼ éå®çå¾çåéæºç3å¼ å¾çè¿è¡éæºè£åªï¼åæ¼æ¥å°ä¸å¼ å¾ä¸ä½ä¸ºè®­ç»æ°æ® ,å¯è§æ¬æ *å¾4.2*ã

è¿æ ·å¯ä»¥ä¸°å¯å¾ççèæ¯ï¼èä¸åå¼ å¾çæ¼æ¥å¨ä¸èµ·åç¸çæé«äº $batch-size$ å¤§å°ï¼åæ¶å¨è¿è¡ $batch \ normalization$ï¼æ¹éå½ä¸åï¼çæ¶åä¹ä¼è®¡ç®åå¼ å¾çã


è¿ä¸­æ¹å¼è½ä½¿å¾ $YOLOv5$ å¯¹äº $batch-size$ å¤§å°å¯¹æ¨¡åè®­ç»ç²¾åº¦çå½±åã


ä¸é¢å¯¹[utils/dataloaders.pyä¸­Mosaic](https://github.com/Oneflow-Inc/one-yolov5/blob/ef218b95d4f6780b3a1d092f7fdc64fd447c9674/utils/dataloaders.py#L764-L832)çå®ç°è¿è¡è§£è¯»ã

```python
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic 

        """
        @param index:  éè¦è·åçå¾åç´¢å¼
        @return: img4: mosaicåä»¿å°å¢å¼ºåçä¸å¼ å¾ç
              labels4: img4å¯¹åºçtarget
        """
        labels4, segments4 = [], []
        # è·åå¾åå°ºå¯¸
        s = self.img_size
        # è¿éæ¯éæºçæmosaicä¸­å¿ç¹
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        # éæºçæå¦å¤3å¼ å¾ççç´¢å¼
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        # å¯¹è¿äºç´¢å¼å¼éæºæåº
        random.shuffle(indices)
        # éåè¿4å¼ å¾ç

        for i, index in enumerate(indices):
            # Load image
            # å è½½å¾çå¹¶è¿åé«å®½

            img, _, (h, w) = self.load_image(index)

            # place img in img4 æ¾ç½®å¾ç
            if i == 0:  # top left(å·¦ä¸è§)
                # çæèæ¯å¾ np.full()å½æ°å¡«ååå§åå¤§å¾ï¼å°ºå¯¸æ¯4å¼ å¾é£ä¹å¤§
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # è®¾ç½®å¤§å¾ä¸çä½ç½®ï¼è¦ä¹åå¾å¤§å°ï¼è¦ä¹æ¾å¤§ï¼ï¼wï¼hï¼æï¼xcï¼ycï¼ï¼æ°çæçé£å¼ å¤§å¾ï¼
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # éåå°å¾ä¸çä½ç½®ï¼åå¾ï¼
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right(å³ä¸è§)
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left(å·¦ä¸è§)
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right(å³ä¸è§)
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            #   å¤§å¾ä¸è´´ä¸å¯¹åºçå°å¾
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # è®¡ç®å°å¾å°å¤§å¾ä¸æ¶æäº§ççåç§»ï¼ç¨æ¥è®¡ç®mosaicå¢å¼ºåçæ ç­¾çä½ç½®
            # Labels  è·åæ ç­¾
            """
            å¯¹labelæ æ³¨è¿è¡åå§åæä½ï¼
            åè¯»åå¯¹åºå¾ççlabelï¼ç¶åå°xywhæ ¼å¼çlabelæ ååä¸ºåç´ xyæ ¼å¼çã
            segments4è½¬ä¸ºåç´ æ®µæ ¼å¼
            ç¶åç»ç»å¡«è¿ä¹ååå¤çæ æ³¨åè¡¨
            """
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # å°xywhï¼ç¾åæ¯é£äºå¼ï¼æ ååä¸ºåç´ xyæ ¼å¼
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                # è½¬ä¸ºåç´ æ®µ
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels æ¼æ¥
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            # np.clipæªåå½æ°ï¼åºå®å¼å¨0å°2så
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()  
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # è¿è¡mosaicçæ¶åå°åå¼ å¾çæ´åå°ä¸èµ·ä¹åshapeä¸º[2*img_size,2*img_size]
        # å¯¹mosaicæ´åçå¾çè¿è¡éæºæè½¬ãå¹³ç§»ãç¼©æ¾ãè£åªï¼å¹¶resizeä¸ºè¾å¥å¤§å°img_size
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove
```
## éä»¶

è¡¨4.1</u>:æ°æ®å¢å¼ºåæ°è¡¨ 

| åæ°å       | éç½®  | è§£æ                                                       |
| ------------ | ----- | ---------------------------------------------------------- |
| hsv_h:       | 0.015 | # image HSV-Hue augmentation (fraction) è²è°               |
| hsv_s:       | 0.7   | # image HSV-Saturation augmentation (fraction) é¥±ååº¦      |
| hsv_v:       | 0.4   | # image HSV-Value augmentation (fraction) æååº¦           |
| degrees:     | 0.0   | # image rotation (+/- deg) æè½¬                            |
| translate:   | 0.1   | # image translation (+/- fraction) å¹³ç§»                    |
| scale:       | 0.9   | # image scale (+/- gain) ç¼©æ¾                              |
| shear:       | 0.0   | # image shear (+/- deg) éå(éåç´æå½±)                   |
| perspective: | 0.0   | # image perspective (+/- fraction), range 0-0.001 éè§åæ¢ |
| flipud:      | 0.0   | # image flip up-down (probability) ä¸ä¸ç¿»è½¬                |
| fliplr:      | 0.5   | # image flip left-right (probability)å·¦å³ç¿»è½¬              |
| mosaic:      | 1.0   | # image mosaic (probability) å¾æ¼æ¥                        |
| mixup:       | 0.1   | # image mixup (probability) å¾åèå                       |
| copy_paste:  | 0.0   | # segment copy-paste (probability) åå²å¡«è¡¥                |




## åèæç« 
- https://docs.ultralytics.com/FAQ/augmentation/