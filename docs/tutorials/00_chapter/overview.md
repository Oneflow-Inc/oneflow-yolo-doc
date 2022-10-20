# 0x0 动机

为了说明使用 OneFlow 训练目标检测模型的可行性以及性能的优越性，最近我们将 ultralytics 版 YOLOv5（https://github.com/ultralytics/yolov5）通过import oneflow as torch的方式迁移为了OneFlow后端（对应YOLOv5的commit号为：`48a85314bc80d8023c99bfb114cea98d71dd0591`）。并对 YOLOv5 中相关的教程进行了汉化，添加了一系列详细的代码解读，原理讲解以及部署教程，希望使得 YOLOv5 项目对用户更加透明化。另外我们也将在性能这个角度进行深入探索，本次我们发布的OneFlow后端的YOLOv5只是一个基础版本，没有用上任何的优化技巧，但即使这样我们在 FP32，3080Ti，CUDA11.7 的条件下在 COCO 上进行训练也比 ultralytics 版 YOLOv5 要快5%-10%左右。相信在后续的一些优化下（比如nn.Graph加持，算子的优化），我们可以继续提升YOLOv5在COCO等数据集的训练速度，更有效缩短目标检测模型的训练时间。

- 🎉代码仓库地址：https://github.com/Oneflow-Inc/one-yolov5
- 🎉文档网站地址：https://start.oneflow.org/oneflow-yolo-doc/index.html

不过即使你对 OneFlow 带来的性能提升不太感兴趣，我们相信[文档网站](https://start.oneflow.org/oneflow-yolo-doc/index.html)中对 YOLOv5 教程的汉化以及源码剖析也会是从零开始深入学习 YOLOv5 一份不错的资料。欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟

欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取最新的动态。


# 0x1. 差异

我们将 YOLOv5 的后端从 PyTorch 换成 OneFlow 之后除了性能优势外还做了一些差异化的内容，其中一些内容已经完成，还有一些正在进行中，下面简单展示一下：

![](https://user-images.githubusercontent.com/35585791/196579121-76c6246e-5793-491e-bf96-86dd5ce06290.png)

##### 🌟1. YOLOv5 网络结构解析 

 文章🎉$1.1$  [YOLOv5 网络结构解析](https://start.oneflow.org/oneflow-yolo-doc/tutorials/01_chapter/yolov5_network_structure_analysis.html)

##### 🌟2. 如何准备yolov5模型训练数据    

文章🎉$2.1$ [如何准备yolov5模型训练数据](https://start.oneflow.org/oneflow-yolo-doc/tutorials/02_chapter/how_to_prepare_yolov5_training_data.html)

##### 🌟3. Model Train(以coco数据集为例)

文章🎉$3.1$ [模型训练](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/model_train.html)

##### 🌟4. YOLOv5的数据组织与处理源码解读
文章🎉$4.1$ [数据增强](https://start.oneflow.org/oneflow-yolo-doc/tutorials/04_chapter/mosaic.html)

##### 🌟5. YOLOv5中Loss部分计算

文章🎉$5.1$ [矩形推理](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/rectangular_reasoning.html)

文章🎉$5.2$ [IOU深入解析](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/iou_in-depth_analysis.html)

文章🎉$5.3$ [模型精确度评估](https://start.oneflow.org/oneflow-yolo-doc/tutorials/05_chapter/map_analysis.html)

施工中...

##### 🌟6. 模型导出和部署介绍

文章🎉$6.1$ [模型导出](https://start.oneflow.org/oneflow-yolo-doc/tutorials/06_chapter/export_onnx_tflite_tensorrt.html)

施工中...

##### 🌟7. 网页部署和app。

施工中...

##### 🌟8. 和tvm的交互，基于tvm的部署。

施工中...

##### 🌟9. YOLOv5中的参数搜索

施工中...

##### 🌟10. oneflow_utils/ 文件夹下的其它trick介绍。

施工中...

这一系列的文章我们将逐步开发，Review 以及发布并且会有相应的视频讲解，我们将这个系列的文章叫作：**《YOLOv5全面解析教程》** 🎉🎉🎉

# 0x2. 在COCO上的精度表现

施工中...

# 0x3. 在COCO上的性能表现

以下的性能结果都是直接将 PyTorch 切换为 OneFlow 之后测试的，**并没有做针对性优化**，后续我们将继续提升 OneFlow 后端 YOLOv5 的训练速度，感兴趣的小伙伴可以关注一下。
## 3080Ti 性能测试结果

### 单卡测试结果
- 以下为GTX 3080ti(12GB) 的yolov5测试结果（oneflow后端 vs PyTorch后端）
- 以下测试结果的数据配置均为coco.yaml，模型配置也完全一样，并记录训练完coco数据集1个epoch需要的时间
- 由于oneflow eager目前amp的支持还不完善，所以我们提供的结果均为fp32模式下进行训练的性能结果
- PyTorch版本 yolov5 code base链接：https://github.com/ultralytics/yolov5
- OneFlow版本 yolov5 code base链接：https://github.com/Oneflow-Inc/one-yolov5
- cuda 版本 11.7, cudnn 版本为 8.5.0
- 测试的命令为：`python train.py --batch 16 --cfg models/yolov5n.yaml --weights '' --data coco.yaml --img 640 --device 0`


![图片](https://user-images.githubusercontent.com/35585791/196843664-ceaabc3c-aae9-40dc-9972-60254f8b2549.png)

在 batch 比较小的时候 OneFlow 后端的 YOLOv5 相比于 PyTorch 有 10% 以上的性能优势，这可能得益于 OneFlow 运行时系统可以更快的做 Kernel Launch。而 batch 比较大的时候 OneFlow 后端的 YOLOv5 相比于 PyTorch 只是稍有优势，这可能是因为当 Batch 比较大的时候 Kernel Launch 的开销占整个运行时间的比例会比较小。

### 两卡DDP测试结果

- 配置和单卡均一致
- 测试的命令为：`python -m oneflow.distributed.launch --nproc_per_node 2 train.py --batch 16 --data coco.yaml --weights '' --device 0,1`

![图片](https://user-images.githubusercontent.com/35585791/196844299-3f6c169d-4606-4e94-9edb-95c1c8935234.png)

得益于单卡的性能优势，在 2 卡DDP模式下，OneFlow 后端的 YOLOv5 的训练时间也是稍微领先 PyTorch 后端的 YoloV5 。



## A100

施工中...

