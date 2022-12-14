>ðä»£ç ä»åºå°åï¼<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
æ¬¢è¿star [one-yolov5é¡¹ç®](https://github.com/Oneflow-Inc/one-yolov5) è·å<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >ææ°çå¨æã</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >å¦ææ¨æé®é¢ï¼æ¬¢è¿å¨ä»åºç»æä»¬æåºå®è´µçæè§ãððð</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
å¦æå¯¹æ¨æå¸®å©ï¼æ¬¢è¿æ¥ç»æStaråð~  </a>

## æ¨¡åå¯¼åº


ð è¿ä¸ªæç¨ç¨æ¥è§£éå¦ä½å¯¼åºä¸ä¸ªè®­ç»å¥½ç OneFlow YOLOv5 æ¨¡å ð  å° ONNX .

### å¼å§ä¹å

åéå·¥ç¨å¹¶å¨ [Python>3.7.0](https://www.python.org/) çç¯å¢ä¸­å®è£ [requiresments.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow è¯·éæ© [nightly çæ¬æè >0.9 çæ¬](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) ã[æ¨¡å](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)å[æ°æ®](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)å¯ä»¥ä»æºç ä¸­èªå¨ä¸è½½ã

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5.git
cd one-yolov5
pip install -r requirements.txt  # install
```

### æ ¼å¼

YOLOv5æ¯æå¤ç§æ¨¡åæ ¼å¼çå¯¼åºï¼å¹¶åºäºç¹å®æ¨¡åå¯¹åºçæ¡æ¶è·å¾æ¨çå éã

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
OneFlow                     | -                             | yolov5s_oneflow_model/
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/

## å¯¼åºè®­ç»å¥½ç YOLOv5 æ¨¡å

ä¸é¢çå½ä»¤æé¢è®­ç»ç YOLOV5s æ¨¡åå¯¼åºä¸º ONNX æ ¼å¼ã`yolov5s` æ¯å°æ¨¡åï¼æ¯å¯ç¨çæ¨¡åéé¢ç¬¬äºå°çãå¶å®éé¡¹æ¯ `yolov5n` ï¼`yolov5m`ï¼`yolov5l`ï¼`yolov5x` ï¼ä»¥åä»ä»¬ç P6 å¯¹åºé¡¹æ¯å¦ `yolov5s6` ï¼æèä½ èªå®ä¹çæ¨¡åï¼å³ `runs/exp/weights/best` ãæå³å¯ç¨æ¨¡åçæ´å¤ä¿¡æ¯ï¼å¯ä»¥åèæä»¬ç[README](https://github.com/Oneflow-Inc/one-yolov5/blob/main/README.md)

```shell
python export.py --weights ../yolov5s/ --include onnx
```

ð¡ æç¤º: æ·»å  --half ä»¥ FP16 åç²¾åº¦å¯¼åºæ¨¡åä»¥å®ç°æ´å°çæä»¶å¤§å°ã

è¾åºï¼

```shell
export: data=data/coco128.yaml, weights=['../yolov5s/'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 ð 270ac92 Python-3.8.11 oneflow-0.8.1+cu117.git.0c70a3f6be CPU

Fusing layers... 
YOLOv5s summary: 157 layers, 7225885 parameters, 229245 gradients

OneFlow: starting from ../yolov5s with output shape (1, 25200, 85) (112.9 MB)

ONNX: starting export with onnx 1.12.0...
Converting model to onnx....
Using opset <onnx, 12>
Optimizing ONNX model
After optimization: Const +17 (73->90), Identity -1 (1->0), Unsqueeze -60 (60->0), output -1 (1->0), variable -60 (127->67)
Succeed converting model, save model to ../yolov5s.onnx
<class 'tuple'>
Comparing result between oneflow and onnx....
Compare succeed!
ONNX: export success, saved as ../yolov5s.onnx (28.0 MB)

Export complete (24.02s)
Results saved to /home/zhangxiaoyu
Detect:          python detect.py --weights ../yolov5s.onnx 
Validate:        python val.py --weights ../yolov5s.onnx 
OneFlow Hub:     model = flow.hub.load('OneFlow-Inc/one-yolov5', 'custom', '../yolov5s.onnx')
Visualize:       https://netron.app
```

å¯¼åºç onnx æ¨¡åä½¿ç¨ [Netron Viewer](https://github.com/lutzroeder/netron) è¿è¡å¯è§åçç»æå¦ä¸ï¼

<img width="1311" alt="å¾ç" src="https://user-images.githubusercontent.com/35585791/196328819-7688631c-f276-444e-a9f4-33079f1d5f98.png">

### å¯¼åºæ¨¡åçç¤ºä¾ç¨æ³

`detect.py` å¯ä»¥å¯¹å¯¼åºçæ¨¡åè¿è¡æ¨çï¼

```python
python path/to/detect.py --weights yolov5s/                  # OneFlow
                                   yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                   yolov5s.xml                # OpenVINO
                                   yolov5s.engine             # TensorRT
                                   yolov5s.mlmodel            # CoreML (macOS only)
                                   yolov5s_saved_model        # TensorFlow SavedModel
                                   yolov5s.pb                 # TensorFlow GraphDef
                                   yolov5s.tflite             # TensorFlow Lite
                                   yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
```

`val.py` å¯ä»¥å¯¹å¯¼åºçæ¨¡åè¿è¡éªè¯ï¼

```python
python path/to/val.py --weights    yolov5s/                  # OneFlow
                                   yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                   yolov5s.xml                # OpenVINO
                                   yolov5s.engine             # TensorRT
                                   yolov5s.mlmodel            # CoreML (macOS only)
                                   yolov5s_saved_model        # TensorFlow SavedModel
                                   yolov5s.pb                 # TensorFlow GraphDef
                                   yolov5s.tflite             # TensorFlow Lite
                                   yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
```

### ONNX Runtime æ¨ç

åºäº onnx æ¨¡åä½¿ç¨ onnxruntime è¿è¡æ¨çï¼

```
python3 detect.py --weights ../yolov5s/yolov5s.onnx 
```

è¾åºï¼

```
detect: weights=['../yolov5s/yolov5s.onnx'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 ð 270ac92 Python-3.8.11 oneflow-0.8.1+cu117.git.0c70a3f6be 
Loading ../yolov5s/yolov5s.onnx for ONNX Runtime inference...
detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 1/2 /home/zhangxiaoyu/one-yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, Done. (0.009s)
image 2/2 /home/zhangxiaoyu/one-yolov5/data/images/zidane.jpg: 640x640 2 persons, 2 ties, Done. (0.011s)
0.5ms pre-process, 10.4ms inference, 4.8ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp14
```

![å¾ç](https://user-images.githubusercontent.com/35585791/196388081-6b6d19c5-c0c5-4c59-9a2b-04f6e37f3c14.png)

### åèæç« 

https://github.com/ultralytics/yolov5/issues/251

