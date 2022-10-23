📚 这个教程用来解释如何从 OneFlow Hub 加载 one-yolov5 。🚀

### 开始之前

在 [Python>3.7.0](https://www.python.org/) 的环境中安装 [所需的依赖库](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt) , OneFlow 请选择 [nightly 版本或者 >0.9 版本](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package) 。[模型](https://github.com/Oneflow-Inc/one-yolov5/tree/main/models)和[数据](https://github.com/Oneflow-Inc/one-yolov5/tree/main/data)可以从源码中自动下载。

💡 专家提示：不需要克隆 https://github.com/Oneflow-Inc/one-yolov5 

### 使用 OneFlow Hub 加载 one-yolov5 

#### 简单的例子

此示例从 OneFlow Hub 加载预训练的 YOLOv5s 模型作为 `model` ，并传一张图像进行推理。 `yolov5s` 是最轻、最快的 YOLOv5 模型。 有关所有可用模型的详细信息，请参阅 [README](https://github.com/Oneflow-Inc/one-yolov5#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A3%80%E6%9F%A5%E7%82%B9) 。

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
print(results.pandas().xyxy[0])


         xmin        ymin         xmax        ymax  confidence  class    name
0  743.290649   48.343842  1141.756348  720.000000    0.879861      0  person
1  441.989624  437.336670   496.585083  710.036255    0.675118     27     tie
2  123.051117  193.237976   714.690674  719.771362    0.666694      0  person
3  978.989807  313.579468  1025.302856  415.526184    0.261517     27     tie
```

#### 更细节的例子

这个例子展示了使用 PIL 和 OpenCV 分别作为图像源的批量推理。`result` 可以打印到控制台，保存到 `runs/hub` , 在支持的环境中显示到屏幕上，并作为张量或 pandas 数据返回。

```python
import cv2
import oneflow as flow
from PIL import Image

# Model
model = flow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s')

# Images
for f in 'zidane.jpg', 'bus.jpg':
    flow.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
im1 = Image.open('zidane.jpg')  # PIL image
im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model([im1, im2], size=640) # batch of images

# Results
results.print()  
results.save()  # or .show()

results.xyxy[0]  # im1 predictions (tensor)
print(results.pandas().xyxy[0])  # im1 predictions (pandas)
```

<center class="half">
    <img src="https://user-images.githubusercontent.com/26833433/124915064-62a49e00-dff1-11eb-86b3-a85b97061afb.jpg" width="400"/><img src="https://user-images.githubusercontent.com/26833433/124915055-60424400-dff1-11eb-9055-24585b375a29.jpg" width="200"/><img src="图片链接" width="200"/>
</center>

对于所有推理选项，请参阅 [YOLOv5 `AutoShape()` forward方法](https://github.com/Oneflow-Inc/one-yolov5/blob/main/models/common.py#L566)。

#### 推理设置

YOLOv5 模型包含各种推理属性，例如置信度阈值、IoU 阈值等，可以通过以下方式设置：

```python
model.conf = 0.25  # NMS confidence threshold
      iou = 0.45  # NMS IoU threshold
      agnostic = False  # NMS class-agnostic
      multi_label = False  # NMS multiple labels per box
      classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
      max_det = 1000  # maximum number of detections per image
      amp = False  # Automatic Mixed Precision (AMP) inference

results = model(im, size=320)  # custom inference size
```

#### 设备

模型创建后可以迁移到任意设备上

```python
model.cpu()  # CPU
model.cuda()  # GPU
model.to(device)  # i.e. device=flow.device(0)
```

模型也可以在任意 `device` 上直接创建：

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', device='cpu') # load on CPU
```

💡 专家提示： 在推理之前，输入图像也会自动传输到模型所在的设备上。

#### 静音输出

使用 `_verbose=False` ,模型可以被静音的加载：

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', _verbose=False)  # load silently
```

#### 输入通道

```python
model = flow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', channels=4)
```

在这种情况下，模型除了第一个输入层外将由预训练的权重组成，它不再与预训练的输入层具有相同的形状。 输入层将保持由随机权重初始化。

#### 类别数

要加载具有 10 个输出类而不是默认的 80 个输出类的预训练 YOLOv5s 模型：

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', classes=10)
```

在这种情况下，模型除了输出层将由预训练的权重组成，它们不再与预训练的输出层具有相同的形状。 输出层将保持由随机权重初始化。

#### 强制重新加载

如果您在上述步骤中遇到问题，设置 `force_reload=True` 可能有助于丢弃现有缓存并强制从 OneFlow Hub 重新下载最新的 YOLOv5 版本。

#### 截图推理

要在桌面屏幕上运行推理：

```python

import oneflow as flow

from PIL import ImageGrab

# Model
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', _verbose=False)

# Image
im = ImageGrab.grab()  # take a screenshot

# Inference
results = model(im)
```

#### 多 GPU 推理

YOLOv5 模型可以加载到多个 GPU 实现多线程推理：

```python
import oneflow as flow
import threading

def run(model, im):
  results = model(im)
  results.save()

# Models
model0 = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', device=0)
model1 = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', device=1)

# Inference
threading.Thread(target=run, args=[model0, 'https://ultralytics.com/images/zidane.jpg'], daemon=True).start()
threading.Thread(target=run, args=[model1, 'https://ultralytics.com/images/bus.jpg'], daemon=True).start()
```

#### 训练
要加载 YOLOv5 模型进行训练而不是推理，请设置 autoshape=False。 要加载具有随机初始化权重的模型（从头开始训练），请使用 pretrained=False。 在这种情况下，您必须提供自己的训练脚本。 或者，请参阅我们的 [YOLOv5 训练自定义数据教程](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/model_train.html)进行模型训练。

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', autoshape=False)  # load pretrained
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'yolov5s', autoshape=False, pretrained=False)  # load scratch
```

#### Base64 结果

用于 API 服务。 有关详细信息，请参阅 [#2291](https://github.com/ultralytics/yolov5/pull/2291) 和 [Flask REST API](https://github.com/ultralytics/yolov5/tree/master/utils/flask_rest_api) 示例。

```python
results = model(im)  # inference

results.ims # array of original images (as np array) passed to model for inference
results.render()  # updates results.ims with boxes and labels
for im in results.ims:
    buffered = BytesIO()
    im_base64 = Image.fromarray(im)
    im_base64.save(buffered, format="JPEG")
    print(base64.b64encode(buffered.getvalue()).decode('utf-8'))  # base64 encoded image with results
```

#### 裁剪结果

返回的检测结果可以被裁剪：

```python
results = model(im)  # inference
crops = results.crop(save=True)  # cropped detections dictionary
```

#### Pandas 结果

结果可以作为[Pandas DataFrames](https://pandas.pydata.org/)返回：

```python
results = model(im)  # inference
results.pandas().xyxy[0]  # Pandas DataFrame
```

<details>
  <summary>Pandas输出（点击展开）</summary>
  <pre><code> 
    print(results.pandas().xyxy[0])
    xmin        ymin         xmax        ymax  confidence  class    name
    0  743.290649   48.343842  1141.756348  720.000000    0.879861      0  person
    1  441.989624  437.336670   496.585083  710.036255    0.675118     27     tie
    2  123.051117  193.237976   714.690674  719.771362    0.666694      0  person
    3  978.989807  313.579468  1025.302856  415.526184    0.261517     27     tie
  </code></pre>
</details>

#### 排序后的结果

结果可以按列排序，例如从左到右（x轴）对车牌数字检测结果进行排序：

```python
results = model(im)  # inference
results.pandas().xyxy[0].sort_values('xmin')  # sorted left-right
```

#### Box-Cropped 结果

结果可以返回并保存为 detection crops：

```python
results = model(im)  # inference
crops = results.crop(save=True)  # cropped detections dictionary
```

#### JSON 结果

结果一旦使用 `.pandas` 被保存为 pandas 数据格式，就可以再使用 `.to_json()` 方法保存为 JSON 格式。可以使用 `orient` 参数修改 JSON 格式。请查看 pandas 的 `.to_json()` 方法的[文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)了解细节。

```python
results = model(ims)  # inference
results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
```

<details>
  <summary>Json输出（点击展开）</summary>
  <pre><code> 
    [{"xmin":743.2906494141,"ymin":48.3438415527,"xmax":1141.7563476562,"ymax":720.0,"confidence":0.87986058,"class":0,"name":"person"},{"xmin":441.9896240234,"ymin":437.3366699219,"xmax":496.5850830078,"ymax":710.0362548828,"confidence":0.6751183867,"class":27,"name":"tie"},{"xmin":123.0511169434,"ymin":193.2379760742,"xmax":714.6906738281,"ymax":719.7713623047,"confidence":0.6666944027,"class":0,"name":"person"},{"xmin":978.9898071289,"ymin":313.5794677734,"xmax":1025.3028564453,"ymax":415.526184082,"confidence":0.2615173161,"class":27,"name":"tie"}]
  </code></pre>
</details>


#### 自定义模型

这个例子展示使用 OneFlow Hub 加载一个自定义的在VOC数据集上进行训练的20个类别的 YOLOV5s 模型 `best` 。

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'custom', path='path/to/best') # local model
model = oneflow.hub.load('/path/to/one-yolov5', 'custom', path='path/to/best') # local repo
```

#### TensorRT, ONNX 和 OpenVINO 模型

OneFlow Hub 支持对大多数 YOLOv5 导出格式进行推理，包括自定义训练模型。查看 [TFLite, ONNX, CoreML, TensorRT 模型导出教程](https://start.oneflow.org/oneflow-yolo-doc/tutorials/06_chapter/export_onnx_tflite_tensorrt.html) 查看细节。

- 💡 专家提示：在 [GPU benchmarks](https://github.com/ultralytics/yolov5/pull/6963) 上 **TensorRT** 可能比PyTorch快3-5倍。
- 💡 专家提示：在 [CPU benchmarks](https://github.com/ultralytics/yolov5/pull/6613)  上 **ONNX** 和 **OpenVINO** 可能比 PyTorch 快2-3倍。

```python
model = oneflow.hub.load('Oneflow-Inc/one-yolov5', 'custom', path='yolov5s/')  # OneFlow
                                                            'yolov5s.onnx')  # ONNX
                                                            'yolov5s_openvino_model/')  # OpenVINO
                                                            'yolov5s.engine')  # TensorRT
                                                            'yolov5s.mlmodel')  # CoreML (macOS-only)
                                                            'yolov5s.tflite')  # TFLite
```


### 参考文章

- https://github.com/ultralytics/yolov5/issues/36
