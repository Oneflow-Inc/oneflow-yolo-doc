# 0x0 动机

为了说明使用 OneFlow 训练目标检测模型的可行性以及性能的优越性，最近我们将ultralytics版YOLOv5（https://github.com/ultralytics/yolov5）通过import oneflow as torch的方式迁移为了OneFlow后端（对应YOLOv5的commit号为：`48a85314bc80d8023c99bfb114cea98d71dd0591`）。并将YOLOv5中相关的教程进行了汉化，添加了一系列详细的代码解读，原理讲解以及部署教程，希望使得YOLOv5仓库更加透明化。另外我们也在性能这个角度进行深入探索，本次我们发布的OneFlow后端的YOLOv5只是一个基础版本，没有用上任何的优化技巧，但即使这样我们在FP32，3090Ti，CUDA11.7的条件下在COCO上训练也比PyTorch的Eager要快5%-10%左右。相信在后续的一些优化下，我们可以继续提速，提升YOLOv5在COCO等数据集的训练速度，有效缩短目标检测模型的训练时间。

- 🎉代码仓库地址：https://github.com/Oneflow-Inc/one-yolov5
- 🎉文档网站地址：https://start.oneflow.org/oneflow-yolo-doc/index.html

即使你对 OneFlow 带来的性能提升不太感兴趣，我们相信这个文档网站中对 YOLOv5 的汉化教程以及源码剖析也会是从零开始深入学习 YOLOv5 不错的资料。欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟


# 0x1. 差异

我们将 YOLOv5 的后端从 PyTorch 换成 OneFlow 之后做了一些差异化的内容，其中一些内容已经完成，还有一些正在进行中，下面简单展示一下：

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

##### 论文解读 📚
- [yolov1论文解读](https://start.oneflow.org/oneflow-yolo-doc/thesis_interpretation/01_yolo.html)
- [yolov2论文解读](https://start.oneflow.org/oneflow-yolo-doc/thesis_interpretation/02_yolo.html)
- [yolov3论文解读](https://start.oneflow.org/oneflow-yolo-doc/thesis_interpretation/03_yolo.html)
- [yolov4论文解读](https://start.oneflow.org/oneflow-yolo-doc/thesis_interpretation/04_yolo.html)
- [yolov6论文解读](https://start.oneflow.org/oneflow-yolo-doc/thesis_interpretation/06_yolo.html)

# 0x2. 在COCO上的精度表现

# 0x3. 在COCO上的性能表现

