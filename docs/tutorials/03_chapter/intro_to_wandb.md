>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取 <a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" > 最新的动态。 </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  > 如果你有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟 </a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对你有帮助，欢迎来给我Star呀😊~  </a>

<img src="https://user-images.githubusercontent.com/109639975/205025624-f1f767f0-efce-4018-82ce-e35777b5c61a.png" width="400" alt="Weights & Biases" />



## 引言
我们可以使用 Weights & Biases(W&B) 来进行机器学习的实验跟踪、数据集版本控制和协作。

<div><img /></div>

<img src="https://user-images.githubusercontent.com/109639975/205025761-f1bb0aea-3b43-484e-9259-7e5966fa8689.png" width="650" alt="Weights & Biases" />

<div><img /></div>

##  仪表盘示例
> 下面是 W&B 中交互式仪表盘的一个示例

![image](https://user-images.githubusercontent.com/109639975/205029427-ae42bb53-926a-49bd-8728-d45de5b954b8.png)


## 数据 & 隐私

W&B 对其云控制仪表盘进行了工业级别的加密。如果您的数据集位于较敏感的环境（如您的企业内部集群），我们推荐使用[on-prem](https://docs.wandb.com/self-hosted)。

下载所有数据并导出到其他工具也很容易，例如，使用Jupyter笔记本进行自定义分析。细节请查阅 W&B 的[API](https://docs.wandb.com/library/api)。


## **Weights & Biases** (W&B) with One-YOLOv5

> 简单两步即可开始记录机器学习实验。


### 1. 安装库

```shell
pip install wandb
```

### 2. 创建账号

注册页注册一个[免费账号](https://wandb.ai/login?signup=true)。

![image](https://user-images.githubusercontent.com/109639975/204803891-9e0bdd4f-05b3-40d4-8b26-f609d8123f2f.png)

终端输入
```shell 
wandb login
```
终端输入后粘贴copy的key 输入回车确认 ，大功告成。

## 验证

> [使用coco128数据集 对 wandb 集成可视化测试结果示例](https://wandb.ai/wearmheart/YOLOv5/runs/3si719qd?workspace=user-wearmheart)

> 在one-yolov5仓库的根目录下

使用指令 ` python train.py --weights ' ' --data data/coco128.yaml --cfg models/yolov5s.yaml `
成功运行示例如下:

![image](https://user-images.githubusercontent.com/109639975/204806938-58fe5e40-b82a-4584-b764-8ea4f2107091.png)

通过W&B: 🚀 View run at：xxx链接即可查看 W&B可视化的结果。

结果报告示例:[使用coco128数据集 对 wandb 集成可视化测试结果](https://wandb.ai/wearmheart/YOLOv5/runs/3si719qd?workspace=user-wearmheart)

## 其他示例

> 使用jupyter-notebook

[创建账户](wandb.ai), 
接着运行以下代码安装"wandb" 包并登录。


```python
!pip install wandb # 安装
import wandb
wandb.login() # 登陆
```


## 可视化实验

> 开始你的第一次可视化训练


1. 开始一个新的训练，并传入超参数以跟踪
2. 记录来自训练或评估的指标
3. 在仪表板中可视化结果


```python
import wandb
import math
import random

# Start a new run, tracking hyperparameters in config
wandb.init(project="test-drive", config={
    "learning_rate": 0.01,
    "dropout": 0.2,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
})
config = wandb.config

# Simulating a training or evaluation loop
for x in range(50):
  acc = math.log(1 + x + random.random()*config.learning_rate) + random.random() + config.dropout
  loss = 10 - math.log(1 + x + random.random() + config.learning_rate*x) + random.random() + config.dropout
  # Log metrics from your script to W&B
  wandb.log({"acc":acc, "loss":loss})

wandb.finish() 
```

![image](https://user-images.githubusercontent.com/109639975/205026937-dad46966-833c-41d7-98b9-7db51ab3b618.png)




