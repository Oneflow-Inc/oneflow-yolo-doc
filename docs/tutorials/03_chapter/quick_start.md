## 安装 📚

```shell
git clone https://github.com/Oneflow-Inc/one-yolov5  # clone
cd one-yolov5
pip install -r requirements.txt  # install
```
## 训练 🚀

### 📌单卡
```shell
$ python train.py  --data coco.yaml --weights yolov5s --device 0
```
### 📌多卡 

```
$ python -m oneflow.distributed.launch --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s --device 0,1
```
注意⚠️：

- --nproc_per_node  指定要使用多少GPU。举个例子🌰:在上面👆 多GPU训练指令中它是2。

- --batch 是总批量大小。它将平均分配给每个GPU。在上面的示例中，每GPU是64/2＝32。
- --cfg : 指定一个包含所有评估参数的配置文件。

- 上面的代码默认使用GPU 0…（N-1）。使用特定的GPU🤔️？
可以通过简单在 --device 后跟指定GPU来实现。「案例🌰」，在下面的代码中，我们将使用GPU 2,3。

```
$ python -m oneflow.distributed.launch --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --device 2,3
```

### 📌恢复训练
如果您的训练进程中断了，您可以这样恢复先前的训练进程。
```shell
# 多卡训练.
python -m oneflow.distributed.launch --nproc_per_node 2 train.py --resume
```
您也可以通过 --resume 参数指定要恢复的模型路径

```shell
# 记得把 /path/to/your/checkpoint/path  替换为您要恢复训练的模型权重路径
--resume /path/to/your/checkpoint/path
```

## 评估 👣

下面的命令是在COCO val2017数据集上以640像素的图像大小测试 `yolov5x` 模型。 `yolov5x`是可用小模型中最大且最精确的，其它可用选项是 `yolov5n` ，`yolov5m`，`yolov5s`，`yolov5l` ，以及他们的 P6 对应项比如 `yolov5s6` ，或者你自定义的模型，即 `runs/exp/weights/best` 。有关可用模型的更多信息，请参阅我们的[README-TABLE](https://github.com/Oneflow-Inc/one-yolov5#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A3%80%E6%9F%A5%E7%82%B9)
```python
$ python val.py --weights yolov5x --data coco.yaml --img 640 
```

## 推理 👍
首先，下载一个训练好的模型权重文件，或选择您自己训练的模型；

然后，
通过 detect.py文件进行推理⚡。

```python
python path/to/detect.py --weights yolov5s --source 0              # webcam
                                                    img.jpg        # image
                                                    vid.mp4        # video
                                                    path/          # directory
                                                    path/*.jpg     # glob
                                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```


<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
快来给我Star呀😊~
  
<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/one-yolo/document/concluding_remarks.gif" align="center">
  
</a>
