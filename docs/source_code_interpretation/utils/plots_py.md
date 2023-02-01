## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>

源码解读： [plots.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/plots.py)



##  1. 导入需要的包和基本配置



```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils 这个脚本都是一些画图工具
"""

import math  # 数学公式模块
from copy import copy  # 提供通用的浅层和深层copy操作
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
from urllib.error import URLError

import cv2  # opencv库
import matplotlib  # matplotlib模块
import matplotlib.pyplot as plt  # matplotlib画图模块
import numpy as np  # numpy矩阵处理函数库
import oneflow as flow  # OneFlow深度学习框架
import pandas as pd  # pandas矩阵操作模块
import seaborn as sn  # 基于matplotlib的图形可视化python包 能够做出各种有吸引力的统计图表
from PIL import Image, ImageDraw, ImageFont  # 图片操作模块

from utils.general import (
    CONFIG_DIR,
    FONT,
    LOGGER,
    Timeout,
    check_font,
    check_requirements,
    clip_coords,
    increment_path,
    is_ascii,
    threaded,
    try_except,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import fitness

# Settings
# 设置一些基本的配置  Settings
RANK = -1  # int(os.getenv('RANK', -1))
matplotlib.rc("font", **{"size": 11})
# 如果这句话放在import matplotlib.pyplot as plt之前就算加上plt.show()也不会再屏幕上绘图 放在之后其实没什么用
matplotlib.use("Agg")  # for writing to files only
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        # self.n 保存颜色个数
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 根据输入的index 选择对应的rgb颜色
        c = self.palette[int(i) % self.n]
        # 返回选择的颜色 默认是rgb
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


# 初始化Colors对象 下面调用colors的时候会调用 __call__函数 (https://www.jianshu.com/p/33d6688246a6)
# __call__: https://docs.python.org/3/reference/datamodel.html?highlight=__call__#emulating-callable-objects
colors = Colors()  # create instance for 'from utils.plots import colors'
```

## 2.save_one_box
> 将预测到的目标从原图中扣出来


```python
def save_one_box(
    xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True
):
    """用在detect.py文件中  由opt的save-crop参数控制是否执行
    将预测到的目标从原图中扣出来 剪切好 并保存 会在 runs/detect/expn 下生成crops文件, 将剪切的图片保存在里面
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    :params xyxy: 预测到的目标框信息 list 4个tensor x1 y1 x2 y2 左上角 + 右下角
    :params im: 原图片 需要裁剪的框从这个原图上裁剪  nparray  (1080, 810, 3)
    :params file: runs\detect\exp\crops\dog\bus.jpg
    :params gain: 1.02 xyxy缩放因子
    :params pad: xyxy pad一点点边界框 裁剪出来会更好看
    :params square: 是否需要将xyxy放缩成正方形
    :params BGR: 保存的图片是BGR还是RGB 。(什么是RGB模式与BGR模式请参阅读 https://blog.csdn.net/SGchi/article/details/104474976)
    :params save: 是否要保存剪切的目标框
    """
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    # view函数: Returns a new tensor with the same data as the self tensor but of a different shape.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=torch+view#torch.Tensor.view
    # 其实按照我们通俗的理解就是reshape，
    # 只不过这里是reshape的是张量，也就是将张量重新调整为自己想要的维度（形状大小）
    xyxy = flow.tensor(xyxy).view(-1, 4)  # list -> Tensor [1, 4] = [x1 y1 x2 y2]
    #   OpenCV中的坐标系定义，如下图所示:
    #   (0,0)o_________width______________x
    #        |                            |
    #        height                       |
    #        |                            |
    #        |                            | 
    #        |                            |
    #        y____________________________o(w,h)
    # xyxy to xywh [1, 4] = [x y w h]
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    # box wh * gain + pad  box*gain再加点pad 裁剪出来框更好看
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # 将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    clip_coords(xyxy, im.shape)
    # crop: 剪切的目标框hw
    crop = im[
        int(xyxy[0, 1]) : int(xyxy[0, 3]),
        int(xyxy[0, 0]) : int(xyxy[0, 2]),
        :: (1 if BGR else -1),
    ]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        # 保存剪切的目标框
        Image.fromarray(crop[..., ::-1],mode="RGB").save(f, quality=95, subsampling=0)  # save RGB
    return crop

```

## 3. plot_results
> 对保存的results.csv日志文件可视化

> 本地日志介绍: https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#_10


```python
def plot_results(file="path/to/results.csv", dir=""):
    """
    :params file: results.csv 文件路径
    :params dir : 本地日志路径， 本地日志介绍: https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#_10
    """
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    # 建造一个figure 分割成2 行5列, 由5个小subplots组成 [Box, Objectness, Classification, P-R, mAP-F1]
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()  # 将多维数组降为一维

    # 遍历每个模糊查询匹配到的results*.csv文件
    files = list(save_dir.glob("results*.csv"))
    assert len(
        files
    ), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)  # 设置子图标题
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()  # 设置子图图例legend
    fig.savefig(save_dir / "results.png", dpi=200)  # 保存result.png
    plt.close()

```

## 4. plot_labels

> 加载数据datasets和labels后 对labels进行可视化 分析labels信息





```python
@try_except  # known issue https://github.com/ultralytics/yolov5/issues/5395
@Timeout(30)  # known issue https://github.com/ultralytics/yolov5/issues/5611
def plot_labels(labels, names=(), save_dir=Path("")):
    # plot dataset labels
    """通常用在train.py中 加载数据datasets和labels后 对labels进行可视化 分析labels信息
    plot dataset labels  生成labels_correlogram.jpg和labels.jpg   画出数据集的labels相关直方图信息
    :params labels: 数据集的全部真实框标签  (num_targets, class+xywh)  
    :params names: 数据集的所有类别名
    :params save_dir: runs\train\exp21
    :params loggers: 日志对象
    """
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    # pd.DataFrame: 创建DataFrame, 类似于一种excel, 表头是['x', 'y', 'width', 'height']  表格数据: b中数据按行依次存储
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    # 1、画出labels的 xywh 各自联合分布直方图  labels_correlogram.jpg
    # seaborn correlogram  seaborn.pairplot  多变量联合分布图: 查看两个或两个以上变量之间两两相互关系的可视化形式
    # data: 联合分布数据x   diag_kind:表示联合分布图中对角线图的类型   kind:表示联合分布图中非对角线图的类型
    # corner: True 表示只显示左下侧 因为左下和右上是重复的   plot_kws,diag_kws: 可以接受字典的参数，对图形进行微调
    sn.pairplot(
        x,
        corner=True,
        diag_kind="auto",
        kind="hist",
        diag_kws=dict(bins=50),
        plot_kws=dict(pmax=0.9),
    )
    # 保存labels_correlogram.jpg
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    # 将整个figure分成2*2四个区域
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    try:  # color histogram bars by class
        [
            y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)
        ]  # known issue #3195
    except Exception:
        pass
    ax[0].set_ylabel("instances")  # 设置y轴label
    if 0 < len(names) < 30:  # 小于30个类别就把所有的类别名作为横坐标
        ax[0].set_xticks(range(len(names)))  # 设置刻度
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)  # 旋转90度 设置每个刻度标签
    else:
        ax[0].set_xlabel("classes")  # 如果类别数大于30个, 可能就放不下去了, 所以只显示x轴label
    # 第三个区域ax[2]画出xy直方图     第四个区域ax[3]画出wh直方图
    sn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    # 第二个区域ax[1]画出所有的真实框
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")  # 不要xy轴

    # 去掉上下左右坐标系(去掉上下左右边框)
    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()
```

labels_correlogram.jpg示例:
    
![image](https://user-images.githubusercontent.com/109639975/204497627-55aca12b-85c9-4a64-abae-cda6bf7c40ae.png)

labels.jpg 示例:

![image](https://user-images.githubusercontent.com/109639975/204499176-f1e60705-f9f1-49eb-ba25-a9ecf0fd9af4.png)



