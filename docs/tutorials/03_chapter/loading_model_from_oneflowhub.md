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