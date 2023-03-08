- [0x0. 引言](#0x0-引言)
- [0x1. 快速开始](#0x1-快速开始)
- [0x2. 在COCO上的精度表现](#0x2-在coco上的精度表现)
  - [yolov5s-default](#yolov5s-default)
  - [yolov5s-seg](#yolov5s-seg)
- [0x3. 在COCO上的单GPU性能表现](#0x3-在coco上的单gpu性能表现)
- [特性 \& bug 修复](#特性--bug-修复)
  - [特性](#特性)
  - [用户反馈的bug](#用户反馈的bug)
- [下个版本的展望](#下个版本的展望)
- [附件](#附件)
- [常用预训练模型下载列表](#常用预训练模型下载列表)


## 0x0. 引言

- 🌟 v1.2.0同步了ultralytics yolov5的上游分支v7.0 ，同时支持分类，目标检测，实例分割任务 
<table border="1px" cellpadding="10px">
        <tr>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220929631-9baf1d12-8cfc-4e9f-985e-372302b672dc.jpg" height="280px"  width="575px"  >
            </td>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220928826-84ed25bc-a72e-46ab-8b9c-c3a2b57ded18.jpg" height="280"  width="575px" >
            </td>
        </tr>
        <tr>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220929320-9f4cf581-43b9-4609-8b51-346c84ac0d62.jpg" height="280"  width="575px" >
            </td>
            <td>
                <img src="https://user-images.githubusercontent.com/109639975/220930143-aa022378-4b6f-4ffc-81bf-3e6032d4862c.jpg" height="280"  width="575px" >
            </td>
        </tr>
        <tr  >
            <td >
                原图 
            </td>
            <td  >
               目标检测: 目标检测是指从图像中检测出多个物体并标记它们的位置和类别。目标检测任务需要给出物体的类别和位置信息，通常使用边界框（bounding box）来表示。目标检测可以应用于自动驾驶、视频监控、人脸识别等领域。
            </td>
        </tr>
        <tr  >
            <td >
               图像分类:  图像分类是指给定一张图像，通过计算机视觉技术来判断它属于哪一类别。
图像分类是一种有监督学习任务，需要通过训练样本和标签来建立分类模型。在图像分类中，算法需要提取图像的特征，然后将其分类为预定义的类别之一。例如，图像分类可以用于识别手写数字、识别动物、区分汽车和自行车等。
            </td>
            <td >
            实例分割: 实例分割是指从图像中检测出多个物体并标记它们的位置和类别，同时对每个物体进行像素级的分割。
实例分割要求更为精细的信息，因为它需要将物体的每个像素都分配给对应的物体。 
实例分割可以应用于医学影像分析、自动驾驶、虚拟现实等领域。
            </td>
        </tr>
    </table>

## 0x1. 快速开始

<details open>
<summary>安装</summary>

在[**Python>=3.7.0**](https://www.python.org/) 的环境中克隆版本仓并安装 [requirements.txt](https://github.com/Oneflow-Inc/one-yolov5/blob/main/requirements.txt)，包括 [OneFlow nightly 或者 oneflow>=0.9.0](https://docs.oneflow.org/master/index.html) 。


```bash
git clone https://github.com/Oneflow-Inc/one-yolov5  # 克隆
cd one-yolov5
pip install -r requirements.txt  # 安装
```

</details>

- [检测模型训练示例](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#_4)
- [分割和分类模型训练示例](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/model_train.html )
  
## 0x2. 在COCO上的精度表现 

###  yolov5s-default

| 后端    | GPU | metrics/mAP_0.5, | metrics/mAP_0.5:0.95, | wandb                                                                                                        |
| ------- | --- | ---------------- | --------------------- | ------------------------------------------------------------------------------------------------------------ |
| OneFlow | 4   | 37.659           | 56.992                | [One-YOLOv5s-default](https://wandb.ai/wearmheart/YOLOv5/runs/tnd1d0t3?workspace=user-wearmheart)            |
| PyTorch | 1   | 37.65            | 56.663                | [YOLOV5s-default](https://wandb.ai/glenn-jocher/YOLOv5_v70_official/runs/ahutyuwd?workspace=user-wearmheart) |


数据 [results.txt](https://github.com/Oneflow-Inc/one-yolov5/files/10694564/results.txt)
<details>
<summary> 启动指令:</summary>

```shell 
python -m oneflow.distributed.launch --nproc_per_node 4  \ 
train.py --batch-size 128 --data coco.yaml --weights " " --cfg models/yolov5s.yaml --img 640 --epochs 300
```
</details>

###  yolov5s-seg

| 后端    | GPU | mAP_0.5:0.95(B) | mAP_0.5:0.95(M) | wandb 日志                                                                                                            |
| ------- | --- | --------------- | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| OneFlow | 8   | 37.558          | 31.402          | [One-YOLOv5s-seg_v1.2.0](https://wandb.ai/wearmheart/YOLOv5-Segment/runs/tt8v7pnm/overview?workspace=user-wearmheart) |
| PyTorch | 1   | 37.705          | 31.651          | [YOLOV5s-seg](https://wandb.ai/glenn-jocher/YOLOv5_v70_official/runs/3difxxrr/overview?workspace=user-wearmheart)     |

<details>
<summary> OneFlow后端启动指令</summary>

```shell
python -m oneflow.distributed.launch --nproc_per_node  8  \
    segment/train.py \
    --data coco.yaml \
    --weights ' ' \
    --cfg yolov5s-seg.yaml   \
    --img 640  \
    --batch-size 320    \
    --device 0,1,2,4      \
    --epochs 300  \
    --bbox_iou_optim --multi_tensor_optimize 
```

</details>


## 0x3. 在COCO上的单GPU性能表现

| 单卡    | amp   | epoch | gpu | batch | 数据集 | 模型            | time(min) |
| ------- | ----- | ----- | --- | ----- | ------ | --------------- | --------- |
| OneFlow | False | 1     | 1   | 8     | coco   | yolov5s-default | 18:49     |
| PyTorch | False | 1     | 1   | 8     | coco   | yolov5s-default | 21:56     |
| OneFlow | False | 1     | 1   | 16    | coco   | yolov5s-default | 14:34     |
| PyTorch | False | 1     | 1   | 16    | coco   | yolov5s-default | 17:46     |
| OneFlow | False | 1     | 1   | 8     | coco   | yolov5s-seg     | 25:36     |
| PyTorch | False | 1     | 1   | 8     | coco   | yolov5s-seg     | 33:16     |
| OneFlow | False | 1     | 1   | 16    | coco   | yolov5s-seg     | 24:07     |
| PyTorch | False | 1     | 1   | 16    | coco   | yolov5s-seg     | 29:55     |

<details>
<summary> 测试环境</summary>

```shell
- 机器  ( 8GPU  NVIDIA GeForce RTX 3090, 24268MiB)
-  oneflow.__version__= '0.9.1+cu117
- torch.__version__= '1.13.0+cu117'
- export NVIDIA_TF32_OVERRIDE=0  # PyTorch使用FP32训练 


# 测试指令:
# OneFlow后端
python   train.py \
    --batch-size 8 \
    --data coco.yaml \
    --weights ' ' \
    --cfg models/yolov5s.yaml \
    --img 640 \
    --epochs 1  \
    --bbox_iou_optim --multi_tensor_optimize

python segment/train.py \
    --data coco.yaml \
    --weights ' ' \
    --cfg  models/segment/yolov5s-seg.yaml \
    --img 640 \
    --batch-size 8
    --epochs 1 \
    --bbox_iou_optim --multi_tensor_optimize 

# PyTorch后端:
export NVIDIA_TF32_OVERRIDE=0 # 使用fp32
python  \
    train.py \
    --batch-size 8 \
    --data coco.yaml \
    --weights ' ' \
    --cfg models/yolov5s.yaml \
    --img 640 \
    --epochs 1  \

export NVIDIA_TF32_OVERRIDE=0 # 使用fp32
python segment/train.py \
    --data coco.yaml \
    --weights ' ' \
    --cfg  models/segment/yolov5s-seg.yaml \
    --img 640 \
    --epochs 1 \
    --batch-size 8
```

</details>


## 特性 & bug 修复
### 特性
<details open>
    <summary> <b> 01 同时支持分类，目标检测，实例分割任务  </b> </summary>
    <a href="https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/model_train.html"> 分割和分类模型训练示例
    </a>
    <br>
    <a href="https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#_3"> 检测模型训练示例 </a>
</details>

<details open>
    <summary> <b> 02 支持flask_rest_api </b> </summary>
    <a href="https://github.com/Oneflow-Inc/one-yolov5/blob/43f5d2b31aaead795465920869214026d7113c9e/utils/flask_rest_api/README.md"> 使用flask_rest_api示例
    </a>
</details>
<details open>
    <summary> <b> 03 支持使用 wandb 对实验跟踪和可视化功能 </b> </summary>
    <a href="https://wandb.ai/wearmheart/YOLOv5/runs/3si719qd?workspace=user-wearmheart"> 使用coco128数据集 对 wandb 集成可视化测试示例
    </a>
    <br>
    <a href="https://github.com/Oneflow-Inc/one-yolov5/pull/87"> 操作指南 </a>
</details>

<details open>
    <summary> <b> 04 oneflow_hub_support_pilimage </b> </summary>
    <a href="https://github.com/Oneflow-Inc/one-yolov5/pull/67"> 操作指南</a>
</details>

<details open>
    <summary> <b> 05 为每个batch的compute_loss部分减少一次h2d和cpu slice_update操作 </b> </summary>
    <a href="https://github.com/Oneflow-Inc/one-yolov5/pull/62"> pr: optim_slice_update_in_compute_loss</a>
</details>

<details open>
    <summary> <b> 06 优化 bbox_iou 函数和模型滑动平均部分，大幅提升训练性能 </b> </summary>
    <a href="https://mp.weixin.qq.com/s/Qh3JCAaPox3TUB0a6Lb_ug"> 消费级显卡的春天，GTX 3090 YOLOv5s单卡完整训练COCO数据集缩短11.35个小时 </a>
</details>

<details open>
    <summary> <b> 07 兼容FlowFlops，训练时可以展示模型的FLOPs </b> </summary>
    <a href="https://mp.weixin.qq.com/s/vnmLqQsndFtq2rc_Ow5Wjg"> 基于 Flowflops 详解深度学习网络的 FLOPs 和 MACs 计算方案 </a>
</details>

### 用户反馈的bug
> 记录了一些用户反馈的常见问题

1.  出现满屏的误检框， 可能到原因场景太单一，泛化不够 ，更多可见我们关于 [如何准备一个好的数据集介绍](https://start.oneflow.org/oneflow-yolo-doc/tutorials/02_chapter/how_to_prepare_yolov5_training_data.html#_5) 或者导出onnx模型进行部署时代码有错误。


2. 这个应该是让batch维度可以动态 你加了dynamic参数？ 暂时不支持该参数 ， 可以自己编辑onnx模型教程 https://github.com/Oneflow-Inc/one-yolov5/releases/download/v1.2.0_/openmmlab.pptx 

3. 模型导出onnx时，出现 `/tmp/oneflow_model322` 类似报错。oneflow新老版本兼容性问题：因为这个是之前旧版本创建的文件但是没清理，删了就可以解决了。

4. 训练过程loss，map，检测框等可视化 我们适配了[wandb](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/intro_to_wandb.html) 

5. device选择这里因为CUDA_VISIBLE_DEVICES环境变量设置放在import oneflow之后会失败，导致device选择失败了，可以export CUDA_VISIBLE_DEVICES=1 这样子手动控制下。

6. autobatch功能 oneflow这边缺少个memory_reserved api ，我们会尽快补齐这个api，现在还是先手动执行下batch_size

## 下个版本的展望

- [ ] 继续提升one-yolov5单卡模式的训练速度，
- [ ] cpu模式下也支持onnx模型的导出，解决显存比原始yolov5稍高的问题等等，
- [ ] OneFlow 研发的amp train目前已经开发完成正在测试中，下个版本将合并进main分支。
- [ ] autobatch功能 
 

## 附件
## 常用预训练模型下载列表
| Model                                                                              | Size(MB) | Model                                                                                | Size(MB) | Model                                                                                      | Size(MB) | Model                                                                                      | Size(MB) |
| ---------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------ | -------- |
| [yolov5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt) | 3.87MB   | [yolov5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt) | 6.86MB   | [yolov5n-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-cls.pt) | 4.87MB   | [yolov5n-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt) | 4.11MB   |
| [yolov5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) | 14.12MB  | [yolov5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt) | 24.78MB  | [yolov5s-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt) | 10.52MB  | [yolov5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt) | 14.87MB  |
| [yolov5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt) | 40.82MB  | [yolov5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt) | 68.96MB  | [yolov5m-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt) | 24.89MB  | [yolov5m-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt) | 42.36MB  |
| [yolov5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt) | 89.29MB  | [yolov5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt) | 147.36MB | [yolov5l-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt) | 50.88MB  | [yolov5l-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt) | 91.9MB   |
| [yolov5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt) | 166.05MB | [yolov5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt) | 269.62MB | [yolov5x-cls](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-cls.pt) | 92.03MB  | [yolov5x-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt) | 170.01MB |




