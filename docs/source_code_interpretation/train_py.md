## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>


源码解读： [train.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/train.py)

> 这个文件是yolov5的训练脚本。总体上代码流程比较简单的，抓住 数据 + 模型 + 学习率 + 优化器 + 训练这五步即可。





## 1. 导入需要的包和基本配置



```python
import argparse       # 解析命令行参数模块
import math           # 数学公式模块
import os             # 与操作系统进行交互的模块 包含文件路径操作和解析
import random         # 生成随机数模块
import sys            # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time           # 时间模块 更底层
from copy import deepcopy # 深度拷贝模块
from datetime import datetime
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np       # numpy数组操作模块
import oneflow as flow   # OneFlow框架
import oneflow.distributed as dist # 分布式训练模块
import oneflow.nn as nn  # 对oneflow.nn.functional的类的封装 有很多和oneflow.nn.functional相同的函数
import yaml              # 操作yaml文件模块
from oneflow.optim import lr_scheduler  # 学习率模块
from tqdm import tqdm   # 进度条模块

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors

# from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader

from utils.downloads import is_url  # , attempt_download
from utils.general import check_img_size  # check_suffix,
from utils.general import (
    LOGGER,
    check_dataset,
    check_file,
    check_git_status,
    check_requirements,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
    model_save,
)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.oneflow_utils import EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer, smart_resume
from utils.plots import plot_evolve, plot_labels
# 这个 Worker 是这台机器上的第几个 Worker
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 这个 Worker 是全局第几个 Worker
RANK = int(os.getenv("RANK", -1))
# 总共有几个 Worker
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

# Linux 下：
# FILE =  'path/to/one-yolov5/train.py'
# 将'path/to/one-yolov5'加入系统的环境变量  该脚本结束后失效。
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

```

## 2. [parse_opt 函数](https://github.com/Oneflow-Inc/one-yolov5/blob/a681bd5ce5853027d366451861241bb09ef6eabd/train.py#L472-L561)


> 这个函数用于设置opt参数

```
weights: 权重文件
cfg: 模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
data: 数据集配置文件 包括path、train、val、test、nc、names、download等
hyp: 初始超参文件
epochs: 训练轮次
batch-size: 训练批次大小
img-size: 输入网络的图片分辨率大小
resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
nosave: 不保存模型  默认False(保存)      True: only test final epoch
notest: 是否只测试最后一轮 默认False  True: 只测试最后一轮   False: 每轮训练完都测试mAP
workers: dataloader中的最大work数（线程个数）
device: 训练的设备
single-cls: 数据集是否只有一个类别 默认False

rect: 训练集是否采用矩形训练  默认False
noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
evolve: 是否进行超参进化 默认False
multi-scale: 是否使用多尺度训练 默认False
label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
adam: 是否使用adam优化器 默认False(使用SGD)
sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
linear-lr: 是否使用linear lr  线性学习率  默认False 使用cosine lr
cache-image: 是否提前缓存图片到内存cache,以加速训练  默认False
image-weights: 是否使用图片采用策略(selection img to training by class weights) 默认False 不使用
bbox_iou_optim: 优化 compute_loss 中的 bbox_iou 函数

bucket: 谷歌云盘bucket 一般用不到
project: 训练结果保存的根目录 默认是runs/train
name: 训练结果保存的目录 默认是exp  最终: runs/train/exp
exist-ok: 如果文件存在就ok不存在就新建或increment name  默认False(默认文件都是不存在的)
quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
save_period: Log model after every "save_period" epoch    默认-1 不需要log model 信息
artifact_alias: which version of dataset artifact to be stripped  默认lastest  貌似没用到这个参数？
local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式

entity: wandb entity 默认None
upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
bbox_interval: 设置带边界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
```


更多细节[请点这](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#_18)

## 3 main函数

### 3.1 [Checks](https://github.com/Oneflow-Inc/one-yolov5/blob/a681bd5ce5853027d366451861241bb09ef6eabd/train.py#L566-L569)


```python
def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        # 输出所有训练opt参数  train: ...
        print_args(vars(opt))
        # 检查代码版本是否是最新的  github: ...
        check_git_status()
        # 检查requirements.txt所需包是否都满足 requirements: ...
        check_requirements(exclude=["thop"])
```

## 3.2 [Resume](https://github.com/Oneflow-Inc/one-yolov5/blob/a681bd5ce5853027d366451861241bb09ef6eabd/train.py#L571-L603)
> 判断是否使用断点续训resume, 读取参数

使用断点续训 就从`path/to/last`中读取相关参数；不使用断点续训 就从文件中读取相关参数


```python
# 2、判断是否使用断点续训resume, 读取参数
if opt.resume and not (check_wandb_resume(opt) or opt.evolve):  # resume from specified or most recent last
    # 使用断点续训 就从last.pt中读取相关参数
    # 如果resume是str，则表示传入的是模型的路径地址
    # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last
    last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
    opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
    opt_data = opt.data  # original dataset
    if opt_yaml.is_file():
        # 相关的opt参数也要替换成last.pt中的opt参数
        with open(opt_yaml, errors="ignore") as f:
            d = yaml.safe_load(f)
    else:
        d = flow.load(last, map_location="cpu")["opt"]
    opt = argparse.Namespace(**d)  # replace
    opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
    if is_url(opt_data):
        opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
else:
    # 不使用断点续训 就从文件中读取相关参数
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
        check_file(opt.data),
        check_yaml(opt.cfg),
        check_yaml(opt.hyp),
        str(opt.weights),
        str(opt.project),
    )  # checks
    assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
    # 将opt.img_size扩展为[train_img_size, test_img_size]
    if opt.evolve:
        if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
            opt.project = str(ROOT / "runs/evolve")
        opt.exist_ok, opt.resume = (
            opt.resume,
            False,
        )  # pass resume to exist_ok and disable resume
    if opt.name == "cfg":
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    # 根据opt.project生成目录  如: runs/train/exp18
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
```

## 3.3  DDP mode
> DDP mode设置


```python
# 3、DDP mode设置
# 选择设备  cpu/cuda:0
device = select_device(opt.device, batch_size=opt.batch_size)
if LOCAL_RANK != -1:
    msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
    assert not opt.image_weights, f"--image-weights {msg}"
    assert not opt.evolve, f"--evolve {msg}"
    assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
    assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
    assert flow.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
    flow.cuda.set_device(LOCAL_RANK)
    device = flow.device("cuda", LOCAL_RANK)
```

### 3.4Train
> 不使用[进化算法](https://so.csdn.net/so/search?q=%E8%BF%9B%E5%8C%96%E7%AE%97%E6%B3%95&spm=1001.2101.3001.7020) 正常Train


```python
# Train
if not opt.evolve:
    # 如果不进行超参进化 那么就直接调用train()函数，开始训练
    train(opt.hyp, opt, device, callbacks)

```

### 3.5 Evolve hyperparameters (optional)
> [遗传进化算法，边进化边训练](https://github.com/Oneflow-Inc/one-yolov5/blob/a681bd5ce5853027d366451861241bb09ef6eabd/train.py#L625-L713)



```python
# 否则使用超参进化算法(遗传算法) 求出最佳超参 再进行训练
else:
    # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    # 超参进化列表 (突变规模, 最小值, 最大值)
    meta = {
        "lr0": (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
        "lrf": (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
        "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
        "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
        "box": (1, 0.02, 0.2),  # box loss gain
        "cls": (1, 0.2, 4.0),  # cls loss gain
        "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
        "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
        "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
        "iou_t": (0, 0.1, 0.7),  # IoU training threshold
        "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
        "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
        "fl_gamma": (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
        "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
        "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
        "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
        "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
        "perspective": (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": (1, 0.0, 1.0),  # image flip up-down (probability)
        "fliplr": (0, 0.0, 1.0),  # image flip left-right (probability)
        "mosaic": (1, 0.0, 1.0),  # image mixup (probability)
        "mixup": (1, 0.0, 1.0),  # image mixup (probability)
        "copy_paste": (1, 0.0, 1.0),
    }  # segment copy-paste (probability)

    with open(opt.hyp, errors="ignore") as f: # 载入初始超参
        hyp = yaml.safe_load(f)  # load hyps dict
        if "anchors" not in hyp:  # anchors commented in hyp.yaml
            hyp["anchors"] = 3
    opt.noval, opt.nosave, save_dir = (
        True,
        True,
        Path(opt.save_dir),
    )  # only val/save final epoch
    # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    # evolve_yaml 超参进化后文件保存地址
    evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
    if opt.bucket:
        os.system(f"gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}")  # download evolve.csv if exists
    
    """
    使用遗传算法进行参数进化 默认是进化300代
    这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
    如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
    有了每个hyp和每个hyp的权重之后有两种进化方式；
    1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
    2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
    evolve.txt会记录每次进化之后的results+hyp
    每次进化时，hyp会根据之前的results进行从大到小的排序；
    再根据fitness函数计算之前每次进化得到的hyp的权重
    再确定哪一种进化方式，从而进行进化
    """
    for _ in range(opt.evolve):  # generations to evolve
        if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
            # Select parent(s)
            parent = "single"  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
            ng = len(meta)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in meta.items():
            hyp[k] = max(hyp[k], v[1])  # lower limit
            hyp[k] = min(hyp[k], v[2])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        # Train mutation
        results = train(hyp.copy(), opt, device, callbacks)
        callbacks = Callbacks()
        # Write mutation results
        print_mutation(results, hyp.copy(), save_dir, opt.bucket)

    # Plot results
    plot_evolve(evolve_csv)
    LOGGER.info(f"Hyperparameter evolution finished {opt.evolve} generations\n" f"Results saved to {colorstr('bold', save_dir)}\n" f"Usage example: $ python train.py --hyp {evolve_yaml}")

```

## 4 def train(hyp, opt, device, callbacks):  



### 4.1 载入参数



```python
"""
:params hyp: data/hyps/hyp.scratch.yaml   hyp dictionary
:params opt: main中opt参数
:params device: 当前设备
"""
def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    (save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, bbox_iou_optim) = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
        opt.bbox_iou_optim,
    )

```

### 4.2 初始化参数和配置信息



```python
# 保存权重路径 如runs/train/exp18/weights
w = save_dir / "weights"  # weights dir
(w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
last, best = w / "last", w / "best"
```


```python
callbacks.run("on_pretrain_routine_start")

# Directories
w = save_dir / "weights"  # weights dir
(w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
last, best = w / "last", w / "best"

# Hyperparameters 超参
if isinstance(hyp, str):
    with open(hyp, errors="ignore") as f:
        # load hyps dict  加载超参信息
        hyp = yaml.safe_load(f)  # load hyps dict
# 日志输出超参信息 hyperparameters: ...
LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
opt.hyp = hyp.copy()  # for saving hyps to checkpoints

# Save run settings
if not evolve:
    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

# Loggers
data_dict = None
if RANK in {-1, 0}:
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

    # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))

# Config
# 是否需要画图: 所有的labels信息、前三次迭代的barch、训练结果等
plots = not evolve and not opt.noplots  # create plots
cuda = device.type != "cpu"

init_seeds(opt.seed + 1 + RANK, deterministic=True)

data_dict = data_dict or check_dataset(data)  # check if None

train_path, val_path = data_dict["train"], data_dict["val"]
# nc: number of classes  数据集有多少种类别
nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
names = ["item"] if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {data}"  # check
# 当前数据集是否是coco数据集(80个类别) 
is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

```

### 4.3 model 


```python
pretrained = os.path.exists(weights)
# 载入模型
if pretrained:
    # 使用预训练
    # ---------------------------------------------------------#
    # 加载模型及参数
    ckpt = flow.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    # 这里加载模型有两种方式，一种是通过opt.cfg 另一种是通过ckpt['model'].yaml
    # 区别在于是否使用resume 如果使用resume会将opt.cfg设为空，按照ckpt['model'].yaml来创建模型
    # 这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
    # 原因: 保存的模型会保存anchors，有时候用户自定义了anchor之后，再resume，则原来基于coco数据集的anchor会自己覆盖自己设定的anchor
    # 详情参考: https://github.com/ultralytics/yolov5/issues/459
    # 所以下面设置intersect_dicts()就是忽略exclude
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    # 筛选字典中的键值对  把exclude删除
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # 载入模型权重
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
else:
    # 不使用预训练
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
# amp = check_amp(model)  # check AMP
amp = False

# Freeze
# 冻结权重层
# 这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    # NaN to 0 (commented for erratic training results)
    # v.register_hook(lambda x: torch.nan_to_num(x))
    if any(x in k for x in freeze):
        LOGGER.info(f"freezing {k}")
        v.requires_grad = False
```

### 4.4 Optimizer
> 选择优化器


```python
# Optimizer
nbs = 64  # nominal batch size
accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
```

### 4.5 学习率


```python
# Scheduler
if opt.cos_lr:
    # 使用one cycle 学习率  https://arxiv.org/pdf/1803.09820.pdf
    lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
else:
    # 使用线性学习率
    def f(x):
        return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]

    lf = f  # linear
# 实例化 scheduler
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

```

### 4.6 EMA 
> 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。



```python
# EMA
ema = ModelEMA(model) if RANK in {-1, 0} else None
```

### 4.7 Resume


```python
# Resume
best_fitness, start_epoch = 0.0, 0
if pretrained:
    if resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
    del ckpt, csd
```

### 4.8 SyncBatchNorm
> [SyncBatchNorm](https://start.oneflow.org/oneflow-yolo-doc/tutorials/03_chapter/quick_start.html#syncbatchnorm)可以提高多gpu训练的准确性，但会显著降低训练速度。它仅适用于多GPU DistributedDataParallel 训练。


```python
# SyncBatchNorm
if opt.sync_bn and cuda and RANK != -1:
    model = flow.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    LOGGER.info("Using SyncBatchNorm()")
```

### 4.9 数据加载


```python
# Trainloader
train_loader, dataset = create_dataloader(
    train_path,
    imgsz,
    batch_size // WORLD_SIZE,
    gs,
    single_cls,
    hyp=hyp,
    augment=True,
    cache=None if opt.cache == "val" else opt.cache,
    rect=opt.rect,
    rank=LOCAL_RANK,
    workers=workers,
    image_weights=opt.image_weights,
    quad=opt.quad,
    prefix=colorstr("train: "),
    shuffle=True,
)
labels = np.concatenate(dataset.labels, 0)
# 获取标签中最大类别值，与类别数作比较，如果小于类别数则表示有问题
mlc = int(labels[:, 0].max())  # max label class
assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

# Process 0
if RANK in {-1, 0}:
    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size // WORLD_SIZE * 2,
        gs,
        single_cls,
        hyp=hyp,
        cache=None if noval else opt.cache,
        rect=True,
        rank=-1,
        workers=workers * 2,
        pad=0.5,
        prefix=colorstr("val: "),
    )[0]
    # 如果不使用断点续训
    if not resume:
        if plots:
            plot_labels(labels, names, save_dir)
        # Anchors
        # 计算默认锚框anchor与数据集标签框的高宽比
        # 标签的高h宽w与anchor的高h_a宽h_b的比值 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
        # 如果bpr小于98%，则根据k-mean算法聚类新的锚框
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision

    callbacks.run("on_pretrain_routine_end")
```

### 4.10 DDP mode


```python
# DDP mode
if cuda and RANK != -1:
    model = smart_DDP(model)
```

### 4.11 附加model attributes


```python
# Model attributes
nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp["box"] *= 3 / nl  # scale to layers
hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
hyp["label_smoothing"] = opt.label_smoothing
model.nc = nc  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model
# 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
model.names = names # 获取类别名
```

### 4.12 Start training


```python
# Start training
t0 = time.time()
nb = len(train_loader)  # number of batches
# 获取热身迭代的次数iterations  # number of warmup iterations, max(3 epochs, 1k iterations)
nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
# nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
last_opt_step = -1
# 初始化maps(每个类别的map)和results
maps = np.zeros(nc)  # mAP per class
results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
# 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
scheduler.last_epoch = start_epoch - 1  # do not move
# scaler = flow.cuda.amp.GradScaler(enabled=amp)

stopper, _ = EarlyStopping(patience=opt.patience), False
# 初始化损失函数
compute_loss = ComputeLoss(model, bbox_iou_optim=bbox_iou_optim)  # init loss class
callbacks.run("on_train_start")
# 打印日志信息
LOGGER.info(
    f"Image sizes {imgsz} train, {imgsz} val\n"
    f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
    f"Logging results to {colorstr('bold', save_dir)}\n"
    f"Starting training for {epochs} epochs..."
)

for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
    callbacks.run("on_train_epoch_start")
    model.train()

    # Update image weights (optional, single-GPU only)
    # Update image weights (optional)  并不一定好  默认是False的
    # 如果为True 进行图片采样策略(按数据集各类别权重采样)
    if opt.image_weights:
        # 根据前面初始化的图片采样权重model.class_weights（每个类别的权重 频率高的权重小）以及maps配合每张图片包含的类别数
        # 通过rando.choices生成图片索引indices从而进行采用 （作者自己写的采样策略，效果不一定ok）
        cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
        iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
        dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
    # 初始化训练时打印的平均损失信息
    mloss = flow.zeros(3, device=device)  # mean losses

    if RANK != -1:
        # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
        train_loader.sampler.set_epoch(epoch)
    
    # 进度条，方便展示信息
    pbar = enumerate(train_loader)

    LOGGER.info(("\n" + "%10s" * 7) % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size"))
    if RANK in {-1, 0}:
        # 创建进度条
        pbar = tqdm(pbar, total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")  # progress bar
    
    # 梯度清零
    optimizer.zero_grad()

    for i, (
        imgs,
        targets,
        paths,
        _,
    ) in pbar:  # batch -------------------------------------------------------------
        callbacks.run("on_train_batch_start")
        # ni: 计算当前迭代次数 iteration
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        # 热身训练（前nw次迭代）热身训练迭代的次数iteration范围[1:nw]  选取较小的accumulate，学习率以及momentum,慢慢的训练
        if ni <= nw:
            xi = [0, nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

        # Multi-scale
        # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
        # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                # 下采样
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

        # Forward
        # imgs = flow.FloatTensor(np.ones([16,3,640,640])).cuda()

        pred = model(imgs)  # forward

        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

        if RANK != -1:
            loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
        if opt.quad:
            loss *= 4.0

        # Backward
        # scaler.scale(loss).backward()
        # Backward  反向传播  
        loss.backward()

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        # 模型反向传播accumulate次（iterations）后再根据累计的梯度更新一次参数
        if ni - last_opt_step >= accumulate:
            # optimizer.step   参数更新
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()
            if ema:
                # 当前epoch训练结束  更新ema
                ema.update(model)
            last_opt_step = ni

        # Log
        # 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
        if RANK in {-1, 0}:
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            pbar.set_description(("%10s" * 1 + "%10.4g" * 5) % (f"{epoch}/{epochs - 1}", *mloss, targets.shape[0], imgs.shape[-1]))

        # end batch ----------------------------------------------------------------

    # Scheduler
    lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
    scheduler.step()

    if RANK in {-1, 0}:
        # mAP
        callbacks.run("on_train_epoch_end", epoch=epoch)
        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

        if not noval or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(
                data_dict,
                batch_size=batch_size // WORLD_SIZE * 2,
                imgsz=imgsz,
                half=amp,
                model=ema.ema,
                single_cls=single_cls,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=False,
                callbacks=callbacks,
                compute_loss=compute_loss,
            )
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        # stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss) + list(results) + lr
        callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

        # Save model
        if (not nosave) or (final_epoch and not evolve):  # if save
            # 测试使用的是ema（指数移动平均 对模型的参数做平均）的模型              
            # results: [1] Precision 所有类别的平均precision(最大f1时)
            #          [1] Recall 所有类别的平均recall
            #          [1] map@0.5 所有类别的平均mAP@0.5
            #          [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
            #          [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
            # maps: [80] 所有类别的mAP@0.5:0.95
            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                "model": deepcopy(de_parallel(model)).half(),
                "ema": deepcopy(ema.ema).half(),
                "updates": ema.updates,
                "optimizer": optimizer.state_dict(),
                "wandb_id": loggers.wandb.wandb_run.id if loggers.wandb else None,
                "opt": vars(opt),
                "date": datetime.now().isoformat(),
            }

            # Save last, best and delete
            model_save(ckpt, last)  # flow.save(ckpt, last)
            if best_fitness == fi:
                model_save(ckpt, best)  # flow.save(ckpt, best)

            if opt.save_period > 0 and epoch % opt.save_period == 0:
                print("is ok")
                model_save(ckpt, w / f"epoch{epoch}")  # flow.save(ckpt, w / f"epoch{epoch}")
            del ckpt
            # Write  将测试结果写入result.txt中
            callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

    # end epoch --------------------------------------------------------------------------
# end training ---------------------------------------------------------------------------
```


```python
### 4.13 End 

打印一些信息
(日志: 打印训练时间、plots可视化训练结果results1.png、confusion_matrix.png 以及(‘F1’, ‘PR’, ‘P’, ‘R’)曲线变化 、日志信息)
+ coco评价(只在coco数据集才会运行) + 释放显存
return 

```


```python
if RANK in {-1, 0}:
    LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours")
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
            if f is best:
                LOGGER.info(f"\nValidating {f}...")
                results, _, _ = val.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=attempt_load(f, device).half(),
                    iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    save_json=is_coco,
                    verbose=True,
                    plots=plots,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )  # val best model with plots

    callbacks.run("on_train_end", last, best, plots, epoch, results)

flow.cuda.empty_cache()
return 
```

## 4 run函数
> 支持指令执行这个脚本   封装train接口


```python
def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt
```

## Reference

- [【YOLOV5-5.x 源码解读】train.py](https://blog.csdn.net/qq_38253797/article/details/119733964)

- [Github: Laughing-q/yolov5_annotations](https://github.com/Laughing-q/yolov5_annotations/blob/master/train.py)

- [CSDN Liaojiajia-2020: YOLOv5代码详解（train.py部分）](https://blog.csdn.net/mary_0830/article/details/107076617?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162910543316780264030293%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162910543316780264030293&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-107076617.pc_search_result_control_group&utm_term=yolov5+train.py&spm=1018.2226.3001.4187)