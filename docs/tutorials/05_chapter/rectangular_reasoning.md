>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >ææ°çå¨æã</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >å¦ææ¨æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹æ¨æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

# ç©å½¢æ¨ç
## ä»ç»
å½æä»¬æä¸å¹å¾çéå¥ç½ç»ï¼è¿å¹å¾çå®½åº¦åé«åº¦ä¸ç½ç»éæ±çä¸ä¸è´çæ¶åï¼æä»¬è¯å®éè¦å¯¹å¾çååºä¸äºæ¹åã

ä¸è¬æ¥è¯´æä¸¤ç§å¸¸ç¨çéæ©:(åè®¾ ç½ç»éæ±çå¾çå¤§å°ä¸º32çåæ°,ä¼ å¥çå¾çé«å®½ä¸º 200 x 416 )
1. æ­£æ¹å½¢æ¨ç(square lnference)
æ¯å°å¾çå¡«åä¸ºæ­£æ¹å½¢,å¦ä¸å¾å·¦è¾¹æç¤ºã
2. ç©å½¢æ¨ç(Rectangular Inference)
å¦ä¸å¾å³è¾¹æç¤ºã

![imgs](./rectangular_reasoning_imgs/Inference.png)

åæ: å¯ä»¥çå°ä¸å¾æ­£æ¹å½¢æ¨çå­å¨å¤§éçåä½é¨å,èå³è¾¹çç©å½¢æ¨çææ¾åä½é¨åå°äºå·¦è¾¹å¹¶ä¸å®éè¡¨ç°çç¸æ¯æ­£æ¹å½¢æ¨çè½æ¾èçåå°æ¨çæ¶é´ã

æ¨çè¿ç¨ï¼å°è¾é¿è¾¹è®¾å®ä¸ºç®æ å°ºå¯¸ 416,512â¦ (å¿é¡»æ¯32çåæ°)ï¼ç­è¾¹ææ¯ä¾ç¼©æ¾ï¼åå¯¹ç­è¾¹è¿è¡è¾å°å¡«åä½¿ç­è¾¹æ»¡è¶³32çåæ°ï¼è¯¦ç»è¿ç¨è¯¦è§æºç è§£æã
## æå±
###  ç©å½¢æ¨çæºç è§£æ
å¯¹åºä»åºæä»¶:

https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/augmentations.py#L93-L131

```python

# å¾çç¼©æ¾ï¼ä¿æå¾ççå®½é«æ¯ä¾ï¼å©ä¸çé¨åç¨ç°è²å¡«åã
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    å°å¾çç¼©æ¾è°æ´å°æå®å¤§å°
    @Param img: åå¾ 
    @Param new_shape: ç¼©æ¾åçå¾çå¤§å°
    @Param color: padçé¢è²
    @Param auto: True ä¿è¯ç¼©æ¾åçå¾çä¿æåå¾çæ¯ä¾ å³ å°åå¾æé¿è¾¹ç¼©æ¾å°æå®å¤§å°ï¼åå°åå¾è¾ç­è¾¹æåå¾æ¯ä¾ç¼©æ¾ï¼ä¸ä¼å¤±çï¼
                 False å°åå¾æé¿è¾¹ç¼©æ¾å°æå®å¤§å°ï¼åå°åå¾è¾ç­è¾¹æåå¾æ¯ä¾ç¼©æ¾,æåå°è¾ç­è¾¹ä¸¤è¾¹padæä½ç¼©æ¾å°æé¿è¾¹å¤§å°ï¼ä¸ä¼å¤±çï¼
    @Param scale_fill: True ç®åç²æ´çå°åå¾resizeå°æå®çå¤§å° ç¸å½äºå°±æ¯resize æ²¡æpadæä½ï¼å¤±çï¼
    @Param scale_up: True  å¯¹äºå°äºnew_shapeçåå¾è¿è¡ç¼©æ¾,å¤§äºçä¸å
                     False å¯¹äºå¤§äºnew_shapeçåå¾è¿è¡ç¼©æ¾,å°äºçä¸å
    @return: img: letterboxåçå¾ç 
             ratio: wh ratios 
             (dw, dh): wåhçpad
    """
    # Resize and pad image while meeting stride-multiple constraints
    # åå¾ççé«å®½
    shape = im.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) è®¡ç®ç¼©æ¾å å­
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    """
    ç¼©æ¾(resize)å°è¾å¥å¤§å°img_sizeçæ¶å,å¦ææ²¡æè®¾ç½®ä¸éæ ·çè¯,ååªè¿è¡ä¸éæ · 
    å ä¸ºä¸éæ ·å¾çä¼è®©å¾çæ¨¡ç³,å¯¹è®­ç»ä¸åå¥½ä¸å½±åæ§è½ã
    """
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding è®¡ç®å¡«å
    ratio = r, r  # width, height ratios
    # æ°çæªå¡«åå¤§å°, ä¿è¯ç¼©æ¾åå¾åæ¯ä¾ä¸å
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle è·åæå°ç©å½¢å¡«å
        # è¿éçåä½æä½å¯ä»¥ä¿è¯paddingåçå¾çæ¯32çæ´æ°å(416x416)ï¼å¦ææ¯(512x512)å¯ä»¥ä¿è¯æ¯64çæ´æ°å
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # å¦æscaleFill = True,åä¸è¿è¡å¡«åï¼ç´æ¥resizeæimg_size,ä»»ç±å¾çè¿è¡æä¼¸ååç¼©
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    # è®¡ç®ä¸ä¸å·¦å³å°å¡«å,å³å°paddingåå°ä¸ä¸ï¼å·¦å³ä¸¤ä¾§
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # å°åå¾resizeå°new_unpad 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # ä¸é¢ä¸¤è¡è®¡ç®éè¦å¡«å padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # è®¡ç®ä¸ä¸ä¸¤ä¾§çpadding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) # è®¡ç®å·¦å³ä¸¤ä¾§çpadding
    # è°ç¨cv2.copyMakeBorderå½æ°è¿è¡èæ¯å¡«åã
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
```


