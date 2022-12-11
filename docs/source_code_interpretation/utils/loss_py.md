## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>

源码解读： [loss.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/loss.py)

其中一些常见的损失函数包括：

分类损失(cls_loss)：该损失用于判断模型是否能够准确地识别出图像中的对象，并将其分类到正确的类别中。

置信度损失(obj_loss)：该损失用于衡量模型预测的框（即包含对象的矩形）与真实框之间的差异。

边界框损失(box_loss)：该损失用于衡量模型预测的边界框与真实边界框之间的差异，这有助于确保模型能够准确地定位对象。

这些损失函数在训练模型时被组合使用，以优化模型的性能。通过使用这些损失函数，YOLOv5可以准确地识别图像中的对象，并将其定位到图像中的具体位置。



## 1. 导入需要的包

```python
import oneflow as flow
import oneflow.nn as nn

from utils.metrics import bbox_iou
from utils.oneflow_utils import de_parallel
```

## 2. smooth_BCE
这个函数是一个标签平滑的策略(trick)，是一种在 分类/检测 问题中，防止过拟合的方法。

如果要详细理解这个策略的原理，请参阅博文: 
[《trick 1》Label Smoothing（标签平滑）—— 分类问题中错误标注的一种解决方法.](https://blog.csdn.net/qq_38253797/article/details/116228065)

smooth_BCE函数代码:



```python
# 标签平滑 
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
```

1. 通常会用在分类损失当中，如下ComputeLoss类的__init__函数定义：
```
self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets
```
2.  ComputeLoss类的__call__函数调用：
```
# Classification
if self.nc > 1:  # cls loss (only if multiple classes)
    t = flow.full_like(pcls, self.cn, device=self.device)  # targets

    # t[range(n), tcls[i]] = self.cp
    t[flow.arange(n, device=self.device), tcls[i]] = self.cp

    lcls = lcls + self.BCEcls(pcls, t)  # BCE
```


## 3. BCEBlurWithLogitsLoss
这个函数是BCE函数的一个替代，是yolov5作者的一个实验性的函数，可以自己试试效果。

使用起来直接在ComputeLoss类的__init__函数中替代传统的BCE函数即可：



```python
class BCEBlurWithLogitsLoss(nn.Module):
    """用在ComputeLoss类的__init__函数中
    BCEwithLogitLoss() with reduced missing label effects.
    https://github.com/ultralytics/yolov5/issues/1030
    The idea was to reduce the effects of false positive (missing labels) 就是检测成正样本了 但是检测错了
    """
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = flow.sigmoid(pred)  # prob from logits
        # dx = [-1, 1]  当pred=1 true=0时(网络预测说这里有个obj但是gt说这里没有), dx=1 => alpha_factor=0 => loss=0
        # 这种就是检测成正样本了但是检测错了（false positive）或者missing label的情况 这种情况不应该过多的惩罚->loss=0
        dx = pred - true  # reduce only missing label effects
        # 如果采样绝对值的话 会减轻pred和gt差异过大而造成的影响
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - flow.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()
```

## 4. FocalLoss
FocalLoss损失函数来自 Kaiming He在2017年发表的一篇论文：Focal Loss for Dense Object Detection. 这篇论文设计的主要思路: 希望那些hard examples对损失的贡献变大，使网络更倾向于从这些样本上学习。防止由于easy examples过多，主导整个损失函数。

优点：

解决了one-stage object detection中图片中正负样本（前景和背景）不均衡的问题；
降低简单样本的权重，使损失函数更关注困难样本；
函数公式：

$F L\left(p_{t}\right)=-\alpha_{t}\left(1-p_{t}\right)^{\gamma} \log \left(p_{t}\right)$

$\begin{array}{c}
p_{t} = \left\{\begin{array}{ll}
p & y = 1 \\
1-p & \text { 其他 }
\end{array}\right.
\end{array}$

$\alpha_{t}=\left\{\begin{array}{ll}
\alpha & y=1(\text { 正样本 }) \\
1-\alpha & \text { 其他 }(\text { 负样本 })
\end{array} ; \text { 其中 } \alpha \in[0,1]\right.$

$\begin{array}{c}
\text { 其中 } \alpha_{t} \text { 来协调正负样本之间的平衡， } \gamma \text { 来降低简单样本的权重，使损失函数更关注困难样本。 }
\end{array}$

FocalLoss函数代码：


```python
class FocalLoss(nn.Module):
    """用在代替原本的BCEcls（分类损失）和BCEobj（置信度损失）
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    论文: https://arxiv.org/abs/1708.02002
    https://blog.csdn.net/qq_38253797/article/details/116292496
    TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    """
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()  定义为多分类交叉熵损失函数
        self.gamma = gamma # 参数gamma  用于削弱简单样本对loss的贡献程度
        self.alpha = alpha # 参数alpha  用于平衡正负样本个数不均衡的问题
        self.reduction = loss_fcn.reduction  # self.reduction: 控制FocalLoss损失输出模式 sum/mean/none  默认是Mean
        # focalloss中的BCE函数的reduction='None'  BCE不使用Sum或者Mean 
        # 需要将Focal loss应用于每一个样本之中
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        # 正常BCE的loss:   loss = -log(p_t)
        loss = self.loss_fcn(pred, true) 
        # p_t = flow.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = flow.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma # 这里代表Focal loss中的指数项
        # 返回最终的loss=BCE * 两个参数  (看看公式就行了 和公式一模一样)
        loss = loss * alpha_factor * modulating_factor
        # 最后选择focalloss返回的类型 默认是mean
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
```

这个函数用在代替原本的BCEcls和BCEobj:
```python
# Focal loss
g = h["fl_gamma"]  # focal loss gamma  g=0 代表不用focal loss
if g > 0:
    BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
```


## 5. QFocalLoss

QFocalLoss损失函数来自20年的一篇文章： [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.](https://arxiv.org/abs/2006.04388)

如果对这篇论文感兴趣可以看看大神博客： [大白话 Generalized Focal Loss.](https://zhuanlan.zhihu.com/p/147691786)

公式:
$\mathbf{Q F L}(\sigma)=-|y-\sigma|^{\beta}((1-y) \log (1-\sigma)+y \log (\sigma))$

QFocalLoss函数代码：


```python

class QFocalLoss(nn.Module):
    """用来代替FocalLoss
    QFocalLoss 来自General Focal Loss论文: https://arxiv.org/abs/2006.04388
    Wraps Quality focal loss around existing loss_fcn(), 
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = flow.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = flow.abs(true - pred_prob) ** self.gamma
        loss = loss * (alpha_factor * modulating_factor)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
```

使用 `QFolcalLoss` 直接在 ComputeLoss 类中使用 `QFolcalLoss`替换掉  `FocalLoss` 即可：
(也就是说用 `QFolcalLoss` 替换如下图代码处的`FocalLoss` )
<a href="https://github.com/Oneflow-Inc/one-yolov5/blob/640ac163ee26a8b13bb2e94f348fb3752a250886/utils/loss.py#L110-L111"
blank="targent">  ![image](https://user-images.githubusercontent.com/109639975/199945719-b458bd18-cedb-45bd-badc-dc5abc07ab30.png)
</a>

## 6. ComputeLoss类
### 6.1 __init__函数




```python
    sort_obj_iou = False # 后面筛选置信度损失正样本的时候是否先对iou排序
    # Compute losses
    def __init__(self, model, autobalance=False):
        # 获取模型所在的设备
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        # Define criteria 定义分类损失和置信度损失
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=flow.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=flow.tensor([h["obj_pw"]], device=device))
        # 标签平滑  eps=0代表不做标签平滑-> cp=1 cn=0  eps!=0代表做标签平滑 
        # cp代表正样本的标签值 cn代表负样本的标签值
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # Focal Loss 的超参数 gamma
        if g > 0:
            # g>0 将分类损失和置信度损失(BCE)都换成focalloss损失函数
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        # m: 返回的是模型的3个检测头分别对应产生的3个输出feature map
        m = de_parallel(model).model[-1]  # Detect() module

        """self.balance  用来实现obj,box,cls loss之间权重的平衡
        {3: [4.0, 1.0, 0.4]} 表示有三个layer的输出，第一个layer的weight是4.0，第二个1.0，第三个以此类推。如果有5个layer的输出才
        """
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # 三个检测头的下采样率m.stride: [8, 16, 32]  .index(16): 求出下采样率stride=16的索引
        # 这个参数会用来自动计算更新3个feature map的置信度损失系数self.balance
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            1.0,
            h,
            autobalance,
        )
        self.na = m.na  # number of anchors  每个grid_cell的anchor数量 = 3
        self.nc = m.nc  # number of classes  数据集的总类别 = 80
        self.nl = m.nl  # number of layers  检测头的个数 = 3
        # anchors: [3, 3, 2]  3个feature map 每个feature map上有3个anchor(w,h)
        # 这里的anchor尺寸是相对feature map的
        self.anchors = m.anchors 
        self.device = device

```

### 6.2 build_targets
这个函数是用来为所有GT筛选相应的anchor正样本。

筛选条件是比较GT和anchor的宽比和高比，大于一定的阈值就是负样本，反之正样本。

筛选到的正样本信息（image_index, anchor_index, gridy, gridx），传入__call__函数，

通过这个信息去筛选pred每个grid预测得到的信息，保留对应grid_cell上的正样本。

通过build_targets筛选的GT中的正样本和pred筛选出的对应位置的预测样本进行计算损失。

补充理解：

这个函数的目的是为了每个gt匹配相应的高质量anchor正样本参与损失计算，

j = flow.max(r, 1. / r).max(2)[0] < self.hyp["anchor_t"]这步的比较是为了将gt分配到不同层上去检测，

后面的步骤是为了将确定在这层检测的gt中心坐标，

进而确定这个gt在这层哪个grid cell进行检测。

做到这一步也就做到了为每个gt匹配anchor正样本的目的。



```python
    # ---------------------------------------------------------
    # build_targets函数用于获得在训练时计算loss函数所需要的目标框，也即正样本。与yolov3/v4的不同，yolov5支持跨网格预测。
    # 对于任何一个GT bbox，三个预测特征层上都可能有先验框anchors匹配，所以该函数输出的正样本框比传入的targets （GT框）数目多
    # 具体处理过程:
    # (1)首先通过bbox与当前层anchor做一遍过滤。对于任何一层计算当前bbox与当前层anchor的匹配程度，不采用IoU，而采用shape比例。如果anchor与bbox的宽高比差距大于4，则认为不匹配，此时忽略相应的bbox，即当做背景;
    # (2)根据留下的bbox，在上下左右四个网格四个方向扩增采样（即对bbox计算落在的网格所有anchors都计算loss(并不是直接和GT框比较计算loss) )
    # 注意此时落在网格不再是一个，而是附近的多个，这样就增加了正样本数。
    # yolov5也没有conf分支忽略阈值(ignore_thresh)的操作，而yoloy3/v4有。
    # --------------------------------------------------------

    def build_targets(self, p, targets):
        
        """所有GT筛选相应的anchor正样本
        这里通过
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : targets.shape[314, 6] 
        解析 build_targets(self, p, targets):函数
        Build targets for compute_loss()
        :params p: p[i]的作用只是得到每个feature map的shape
                   预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                   如: list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
                   [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                   可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index
                        gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # na = 3 ; nt = 314
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        
        tcls, tbox, indices, anch = [], [], [], []
        # gain.shape=[7]
        gain = flow.ones(7, device=self.device)  # normalized to gridspace gain
        # ai.shape = (na,nt) 生成anchor索引
        # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个 
        #  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 314]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        # ai.shape  =[3, 314]
        ai = flow.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        
        # [314, 6] [3, 314] -> [3, 314, 6] [3, 314, 1] -> [3, 314, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target
        # targets.shape = [3, 314, 7]
        targets = flow.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        # 设置网格中心偏移量
        g = 0.5  # bias
        # 附近的4个框
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = (
            flow.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets
        # 对每个检测层进行处理 
        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl): #  self.nl: number of detection layers   Detect的个数 = 3
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
            anchors, shape = self.anchors[i], p[i].shape

            # gain: 保存每个输出feature map的宽高 -> gain[2:6] = flow.tensor(shape)[[3, 2, 3, 2]] 
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = flow.tensor(p[i].shape, device=self.device)[[3, 2, 3, 2]].float()  # xyxy gain
            # Match targets to anchors
            # t.shape = [3, 314, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #    [3, 314, image_index+class+xywh+anchor_index]
            t = targets * gain  # shape(3,n,7)
            if nt: # 如果有目标就开始匹配
                # Matches
                # 所有的gt与当前层的三个anchor的宽高比(w/w  h/h)
                # r.shape = [3, 314, 2]
                r = t[..., 4:6] / anchors[:, None]  # wh ratio              
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # flow.max(r, 1. / r)=[3, 314, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j.shape = [3, 314]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                j = flow.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare 
                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 根据筛选条件j, 过滤负样本, 得到所有gt的anchor正样本(batch_size张图片)
                # 知道当前gt的坐标 属于哪张图片 正样本对应的idx 也就得到了当前gt的正样本anchor
                # t: [3, 314, 7] -> [555, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter
                # Offsets
                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  
                # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 
                # 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个
                # 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                # gxy.shape = [555, 2]
                gxy = t[:, 2:4]  # grid xy
                # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # gxi.shape = [555, 2]
                gxi = gain[[2, 3]] - gxy  # inverse
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 
                # 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [555] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [555] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [555] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [555] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j.shape=[5, 555]
                j = flow.stack((flow.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*555 都不在边上等号成立
                # t: [555, 7] -> 复制5份target[5, 555, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 555] + t: [5, 555, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                 # flow.zeros_like(gxy)[None]: [1, 555, 2]   off[:, None]: [5, 1, 2]  => [5, 555, 2]
                # j筛选后: [1659, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (flow.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc.shape = [1659, 2]
            # gxy.shape = [1659, 2]
            # gwh.shape  = [1659, 2]
            # a.shape = [1659, 1]
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # a.shape = [1659]
            # (b,c).shape = [1659, 2]
            a, (b, c) = (
                a.contiguous().long().view(-1),
                bc.contiguous().long().T,
            )  # anchors, image, class

            # gij = (gxy - offsets).long()
            # 预测真实框的网格所在的左上角坐标(有左上右下的网格)  
            # gij.shape = [1659, 2]
            gij = (gxy - offsets).contiguous().long() 
            # gi.shape = [1659]
            # gj.shape = [1659]
            gi, gj = gij.T  # grid indices

            # Append

            # indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # gi.shape = [1659]
            # gj.shape = [1659]
            gi = gi.clamp(0, shape[3] - 1)
            gj = gj.clamp(0, shape[2] - 1)
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj, gi))  # image, anchor, grid
            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(flow.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors   对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
```

### 6.3 __call__函数
这个函数相当于forward函数，在这个函数中进行损失函数的前向传播。


```python
    def __call__(self, p, targets):  # predictions, targets
        """
        这里通过输入
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : targets.shape[314, 6] 
        解析__call__函数

        :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: ([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
                    [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell
                    (每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [314, 6] [num_object,  batch_index+class+xywh]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params flow.cat((lbox, lobj, lcls, loss)).detach():
        回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        # 初始化各个部分损失   始化lcls, lbox, lobj三种损失值  tensor([0.])
        # lcls.shape = [1]
        lcls = flow.zeros(1, device=self.device)  # class loss 
        # lbox.shape = [1]
        lbox = flow.zeros(1, device=self.device)  # box loss
        # lobj.shape = [1]
        lobj = flow.zeros(1, device=self.device)  # object loss
        # 获得标签分类,边框,索引，anchors
        # 每一个都是append的 有feature map个 
        # 都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失) 
        #          gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  
        # 可能一个target会使用大小不同anchor进行计算
        """shape
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : [314, 6]
        tcls    : list([1659], [1625], [921])
        tbox    : list([1659, 4], [1625, 4], [921, 4])
        indices : list( list([1659],[1659],[1659],[1659]), list([1625],[1625],[1625],[1625]) , list([921],[921],[921],[921])  )
        anchors : list([1659, 2], [1625, 2], [921, 2])
        """
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            # 这里通过 pi 形状为[16, 3, 80, 80, 85] 进行解析
            """shape
            b   : [1659]
            a   : [1659]
            gj  : [1659]
            gi  : [1659]
            """
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            # tobj = flow.zeros( pi.shape[:4] , dtype=pi.dtype, device=self.device)  # target obj
            # 初始化target置信度(先全是负样本 后面再筛选正样本赋值)
            # tobj.shape = [16, 3, 80, 80]
            tobj = flow.zeros((pi.shape[:4]), dtype=pi.dtype, device=self.device)  # target obj
            # n = 1659
            n = b.shape[0]  # number of targets
            if n:
                # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires flow 1.8.0
                """shape
                pxy     : [1659, 2]
                pwh     : [1659, 2]
                _       : [1659, 1]
                pcls    : [1659, 80]
                """
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression loss  只计算所有正样本的回归损失
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable
                # pxy.shape = [1659, 2]
                pxy = pxy.sigmoid() * 2 - 0.5
                # https://github.com/ultralytics/yolov3/issues/168
                # pwh.shape = [1659, 2]
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] # 和论文里不同 这里是作者自己提出的公式
                # pbox.shape = [1659, 4]
                pbox = flow.cat((pxy, pwh), 1)  # predicted box
                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练 传回loss 修改梯度 让pbox越来越接近tbox(偏移量)
                # iou.shape = [1659]
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # lbox.shape = [1]
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                # iou.shape = [1659]
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # 这里对iou进行排序在做一个优化：当一个正样本出现多个GT的情况也就是同一个grid中有两个gt(密集型且形状差不多物体)
                # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                if self.sort_obj_iou:
                    # https://github.com/ultralytics/yolov5/issues/3605
                    # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                    j = iou.argsort()
                    # 排序之后 如果同一个grid出现两个gt 那么我们经过排序之后每个grid中的score_iou都能保证是最大的
                    # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后1个总是和最大的IOU去计算LOSS, 梯度传播
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们人为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小置信度越接近1(人为加大训练难度)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification 只计算所有正样本的分类损失 
                # self.nc = 80
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    # t.shape = [1659,80]
                    t = flow.full_like(pcls, self.cn, device=self.device)  # targets

                    # t[range(n), tcls[i]] = self.cp  筛选到的正样本对应位置值是cp 
                
                    t[flow.arange(n, device=self.device), tcls[i]] = self.cp
                    # lcls.shape = [1]
                    lcls = lcls + self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in flow.cat((txy[i], twh[i]), 1)]
            #  置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj = lobj + (obji * self.balance[i])  # obj loss

            if self.autobalance:
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        """shape
        lbox    : [1]
        lobj    : [1]
        lcls    : [1]
        """
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        # loss = lbox + lobj + lcls  平均每张图片的总损失
        # loss * bs: 整个batch的总损失
        # .detach()  利用损失值进行反向传播 利用梯度信息更新的是损失函数的参数 
        # 于损失这个值是不需要梯度反向传播的
        return (lbox + lobj + lcls) * bs, flow.cat((lbox, lobj, lcls)).detach()
```

使用：

1. train.py初始化损失函数类：

<a href="https://github.com/Oneflow-Inc/one-yolov5/blob/640ac163ee26a8b13bb2e94f348fb3752a250886/train.py#L286" blank="target">
 compute_loss = ComputeLoss(model)  # init loss class
</a>


2. 调用执行损失函数，计算损失：

<a href="https://github.com/Oneflow-Inc/one-yolov5/blob/640ac163ee26a8b13bb2e94f348fb3752a250886/train.py#L355" blank="target">     loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size </a>


## 总结

这个脚本最最最重要的就是ComputeLoss类了。看了很久，本来打算写细一点的，但是看完代码发现自己把想说的都已经写在代码的注释当中了。代码其实还是挺难的，尤其build_target各种花里胡哨的矩阵操作较多，pytorch不熟的人会看的比较痛苦，但是如果你坚持看下来我的注释再加上自己的debug的话，应该是能读懂的。最后，一定要细读ComputeLoss！！！！


## Reference
- [【YOLOV5-5.x 源码解读】loss.py](https://blog.csdn.net/qq_38253797/article/details/119444854)
- [目标检测 YOLOv5 - Sample Assignment](https://blog.csdn.net/flyfish1986/article/details/119332396)
- [yolov5--loss.py --v5.0版本-最新代码详细解释-2021-7-1更新](https://blog.csdn.net/qq_21539375/article/details/118345636)

- [YOLO-V3-SPP 训练时正样本筛选源码解析之build_targets](https://blog.csdn.net/qq_38109282/article/details/119411005)

- [YOLOv5-4.0-loss.py 源代码导读(损失函数）](https://blog.csdn.net/weixin_42716570/article/details/116759811)

- [yolov5 代码解读 损失函数 loss.py](https://blog.csdn.net/guikunchen/article/details/118452790)
