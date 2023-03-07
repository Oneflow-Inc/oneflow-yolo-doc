## 实例分割 
> 使用示例 🚀
### Train 
YOLOv5实例分割模型支持使用 `--data coco128-seg.yaml`  参数自动下载 `COCO128-seg` 测试数据集(*测试数据集表示能测试项目正常运行的小数据集*)， 以及使用 `bash data/scripts/get_coco.sh --train --val --segments`  或者使用  `python train.py --data coco.yaml`  下载 `COCO-segments` 数据集

```shell
# Single-GPU
python segment/train.py --model yolov5s-seg.of --data coco128-seg.yaml --epochs 5 --img 640

# Multi-GPU DDP
python -m oneflow.distributed.launch --nproc_per_node  4  segment/train.py --model yolov5s-seg.of --data coco128-seg.yaml --epochs 5 --img 640 --device 0,1,2,3
```

注意 :
- {`.of`: 代表OneFlow预训练权重 , `.pt`: 代表 PyTorch 预训练权重 }
- `--model yolov5s-seg.of`  表示使用OneFlow预训练权重 , 也是支持使用 PyTorch 预训练权重 如 `--model yolov5s-seg.pt`
- 模型权重将自动从 github 下载(*建议如果没有设置代理，可以提前将模型下载到电脑本地 使用 `--model 本地路径/yolov5s-seg.of`*)

### val 

数据集上验证YOLOv5m-seg 模型的精度

```shell 
bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.of --data coco.yaml --img 640  # validate
```

### Predict 

使用预训练模型(YOLOv5m-seg) 预测 

```shell
python segment/predict.py --weights yolov5m-seg.of --data data/images/
```

![image](https://user-images.githubusercontent.com/118866310/223043320-ba3599d9-a3a4-4590-af98-65da1e3f228c.png)

### Export

将 `yolov5s-seg` 模型导出为 ONNX 格式 示例
```shell
python export.py --weights yolov5s-seg.of --include onnx  --img 640 --device 0
```

## 分类
> 使用示例 🚀
### Train 
YOLOv5实例分类模型支持使用 `--data imagenette160`  参数自动下载 `imagenette160` 测试数据集(*测试数据集表示能测试项目正常运行的小数据集*)， 以及使用 `bash data/scripts/get_imagenet.sh`  或者使用  `python train.py --data imagenet`  下载 `imagenet` 数据集

```shell
# Single-GPU
python classify/train.py --model yolov5s-cls.of --data imagenette160 --epochs 5  

# Multi-GPU DDP
python -m oneflow.distributed.launch --nproc_per_node  4  classify/train.py --model yolov5s-cls.of --data imagenette160 --epochs 5   --device 0,1,2,3
```

注意 :
- {`.of`: 代表OneFlow预训练权重 , `.pt`: 代表 PyTorch 预训练权重 }
- `--model yolov5s-cls.of`  表示使用OneFlow预训练权重 , 也是支持使用 PyTorch 预训练权重 如 `--model yolov5s-seg.pt`
- 模型权重将自动从 github 下载(*建议如果没有设置代理，可以提前将模型下载到电脑本地 使用 `--model 本地路径/yolov5s-cls.of`*)

### val 

在ImageNet 数据集上验证YOLOv5m-cls 模型的精度

```shell 
bash data/scripts/get_imagenet.sh  # Download ILSVRC2012 ImageNet dataset https://image-net.org
python classify/val.py --data ../datasets/imagenet --img 224 --weights yolov5s-cls.of
```

### Predict 

使用预训练模型(YOLOv5m-cls) 预测 bus.jpg 

```shell
python classify/predict.py --weights runs/yolov5s-cls.of --source data/images/bus.jpg 
```
![image](https://user-images.githubusercontent.com/118866310/223079567-f9fadd7c-6e76-4f3d-ba2d-a1484e1e5d20.png)

### Export

将 `yolov5s-cls` 模型导出为 ONNX 格式 示例
```shell
python export.py --weights yolov5s-cls.of --include onnx    --device 0
```
