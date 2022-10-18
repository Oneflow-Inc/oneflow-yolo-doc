## 模型导出


📚 这个教程用来解释如何导出一个训练好的 OneFlow YOLOv5 模型 🚀  到 ONNX .

### 开始之前

克隆工程并在 [Python>3.7.0](https://www.python.org/) 的环境中安装 [requiresments.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow 请选择 [nightly 版本或者 >0.9 版本](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) 。[模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)和[数据](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)可以从源码中自动下载。

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5.git
cd one-yolov5
pip install -r requirements.txt  # install
```

### 格式

YOLOv5支持多种模型格式的导出，并基于特定模型对应的框架获得推理加速。

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

## 导出训练好的 YOLOv5 模型

下面的命令把预训练的 YOLOV5s 模型导出为 ONNX 格式。`yolov5s` 是小模型，是可用的模型里面第二小的。其它选项是 `yolov5n` ，`yolov5m`，`yolov5l`，`yolov5x` ，以及他们的 P6 对应项比如 `yolov5s6` ，或者你自定义的模型，即 `runs/exp/weights/best.pt` 。有关可用模型的更多信息，可以参考我们的[README](https://github.com/Oneflow-Inc/one-yolov5/blob/main/README.md)

```shell
python export.py --weights ../yolov5s/ --include onnx
```

💡 提示: 添加 --half 以 FP16 半精度导出模型以实现更小的文件大小。

输出：

```shell
export: data=data/coco128.yaml, weights=['../yolov5s/'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 270ac92 Python-3.8.11 oneflow-0.8.1+cu117.git.0c70a3f6be CPU

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

导出的 onnx 模型使用 [Netron Viewer](https://github.com/lutzroeder/netron) 进行可视化的结果如下：

<img width="1311" alt="图片" src="https://user-images.githubusercontent.com/35585791/196328819-7688631c-f276-444e-a9f4-33079f1d5f98.png">

### 导出模型的示例用法

`detect.py` 可以对导出的模型进行推理：

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

`val.py` 可以对导出的模型进行验证：

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

### OpenCV DNN 推理

基于 onnx 模型使用 onnxruntime 进行推理：

```
python3 detect.py --weights ../yolov5s/yolov5s.onnx 
```

输出：

```
detect: weights=['../yolov5s/yolov5s.onnx'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
YOLOv5 🚀 270ac92 Python-3.8.11 oneflow-0.8.1+cu117.git.0c70a3f6be 
Loading ../yolov5s/yolov5s.onnx for ONNX Runtime inference...
detect.py:159: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
image 1/2 /home/zhangxiaoyu/one-yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, Done. (0.009s)
image 2/2 /home/zhangxiaoyu/one-yolov5/data/images/zidane.jpg: 640x640 2 persons, 2 ties, Done. (0.011s)
0.5ms pre-process, 10.4ms inference, 4.8ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs/detect/exp14
```

![图片](https://user-images.githubusercontent.com/35585791/196388081-6b6d19c5-c0c5-4c59-9a2b-04f6e37f3c14.png)

