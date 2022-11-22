## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>


源码解读： [utils/autoanchor.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/autoanchor.py)

## 摘要
&emsp;**维度聚类**（Dimension Clusters）。当把 YOLO 结合 $anchor \ boxes$ 使用时，我们会遇到两个问题： 首先 $anchor \ boxes$ 的尺寸是手工挑选的。虽然网络可以通过学习适当地调整$anchor \ boxes$ 形状，但是如果我们从一开始就为网络选择更好的 $anchor \ boxes$ ，就可以让网络更容易学习到并获得更好的检测结果。

![image](https://user-images.githubusercontent.com/109639975/199901435-76986df9-cc7b-4eac-97f1-fc905ed3d8d7.png)

图1：$VOC$ 和  $COCO$  上的聚类框尺寸。我们在边界框的维度(dimensions of bounding boxes) 上运行 $k-means$ 聚类，以获得我们模型的良好初始 $anchor \ boxes$ 。左图显示了我们通过 k 的各种选择获得的 $Avg \ IoU$ 。我们发现 $k = 5$ 为召回与模型的复杂性提供了良好的折中。The right image shows the relative centroids for VOC and COCO. (右图显示了在 $VOC$ 和 $COCO$ 上簇的相对中心),并且这两种方案都喜欢更稀疏的，更高的框，此外在 $COCO$ 的尺寸的变化比 $VOC$ 更大。


&emsp;我们不用手工选择 $anchor \ boxes$，而是在训练集的边界框上的维度上运行 k-means聚类算法，自动找到良好的 $anchor \ boxes$ 。 如果我们使用具有欧几里得距离的标准 $k-means$ ，那么较大的框比较小的框产生更多的误差。 然而，我们真正想要的是独立于框的大小的，能获得良好的 $IoU$ 分数的 $anchor \ boxes$ 。 因此对于距离度量我们使用:

<center>

$d(\text { box, centroid }) = 1-\operatorname{IoU}(\text { box }, \text { centroid })$

</center>

&emsp;我们用不同的 $k$ 值运行 $k-means$ 算法，并绘制最接近质心的平均 $Avg \ IoU$（见图1）。为了在模型复杂度和高召回率之间的良好折中，我们选择 $k = 5$ （*也就是5种anchor boxes*）簇的相对中心 与手工选取的 $anchor \ boxes$ 显着不同，它有更少的短且宽的框，并且有更多既长又窄的框。


&emsp;表1中，我们将聚类策略得到的 $anchor \ boxes$ 和手工选取的 $anchor \ boxes$ 在最接近的 $Avg \ IoU$ 上进行比较。通过聚类策略得到的仅5种 $anchor \ boxes$ 中心的 $Avg \ IoU$ 为61.0，其性能类似于9个通过网络学习的 $anchor \ boxes$ 的60.9 (*即Avg IoU已经达到了Faster RCNN的水平*)。 而且使用9种 $anchor \ boxes$ 会得到更高的 $Avg \ IoU$ 。这表明使用 $k-means$ 生成 $anchor \ boxes$ 可以更好地表示模型并使其更容易学习。



$\begin{array}{lcc}
\text { Box Generation } & \# & \text { Avg IoU } \\
\hline \text { Cluster SSE } & 5 & 58.7 \\
\text { Cluster IoU } & 5 & 61.0 \\
\text { Anchor Boxes [15] } & 9 & 60.9 \\
\text { Cluster IoU } & 9 & 67.2
\end{array}$

表1： $VOC \  2007$ 最接近先验的框的 $Avg \ IoU$。 $VOC \  2007$ 上的目标的$Avg \ IoU$与其最接近的，未经修改的使用不同生成方法的目标之间的 $Avg \ IoU$ 。聚类得结果比使用手工选取的 $anchor \ boxes$ 结果要好得多。 

## 什么是k-means?
&emsp;k-means是非常经典且有效的聚类方法，通过计算样本之间的距离（相似程度）将较近的样本聚为同一类别（簇）。

在One-YOLOv5项目中使用K-means
  1. train.py的parse_opt下的参数noautoanchor必须为False 
  2. hpy.scratch.yaml下的anchors参数注释掉。

### 使用k-means时主要关注两点

1. 如何表示样本与样本之间的距离（核心问题），这个一般需要根据具体场景去设计，不同的方法聚类效果也不同，最常见的就是欧式距离，在目标检测领域常见的是IoU。
2. 分为几类，这个也是需要根据应用场景取选择的，也是一个超参数。
### k-means算法主要流程

1. 手动设定簇的个数k，假设k=2；
2. 在所有样本中随机选取k个样本作为簇的初始中心，如下图（random clusters）中两个黄色的小星星代表随机初始化的两个簇中心；
3. 计算每个样本离每个簇中心的距离（这里以欧式距离为例），然后将样本划分到离它最近的簇中。如下图（step 0）用不同的颜色区分不同的簇；
4. 更新簇的中心，计算每个簇中所有样本的均值（方法不唯一）作为新的簇中心。如下图（step 1）所示，两个黄色的小星星已经移动到对应簇的中心；
5. 重复第3步到第4步直到簇中心不在变化或者簇中心变化很小满足给定终止条件。如下图（step2）所示，最终聚类结果。

![image](https://user-images.githubusercontent.com/109639975/200206147-46531a06-5011-4020-ab7c-967ddf9c0df2.png)

### 什么是BPR?
BPR（bpr best possible recall来源于论文: [FCOS](https://arxiv.org/abs/1904.01355).

原论文解释：

> BPR is defined as the ratio of the number of ground-truth boxes a detector can recall at the most divided by all ground-truth boxes. A ground-truth box is considered being recalled if the box is assigned to at least one sample (i.e., a location in FCOS or an anchor box in anchor-based detectors) during training.

&emsp;bpr(best possible recall): 最多能被召回的ground truth框数量 / 所有ground truth框数量 最大值为1 越大越好 小于0.98就需要使用k-means + 遗传进化算法选择出与数据集更匹配的anchor boxes框。

### 什么是白化操作whiten？
&emsp;白化的目的是去除输入数据的冗余信息。假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性，所以用于训练时输入是冗余的；白化的目的就是降低输入的冗余性。

输入数据集X，经过白化处理后，新的数据X’满足两个性质：

1. 特征之间相关性较低；
2. 所有特征具有相同的方差=1

&emsp;常见的作法是：对每一个数据做一个标准差归一化处理（除以标准差）。scipy.cluster.vq.kmeans() 函数输入的数据就是必须是白化后的数据。相应的输出的[anchor](https://so.csdn.net/so/search?q=anchor&spm=1001.2101.3001.7020) k也是白化后的anchor，所以需要将anchor k 都乘以标准差恢复。


## 1. 导入需要的包


```python
import numpy as np      # numpy矩阵操作模块
import oneflow as flow  # OneFlow深度学习模块
import yaml             # 操作yaml文件模块
from tqdm import tqdm   # Python进度条模块

from utils.general import LOGGER, colorstr # 日志模块

PREFIX = colorstr("AutoAnchor: ")
```

## 2.check_anchor_order
这个函数用于确认当前anchors和stride的顺序是否是一致的，因为我们的m.anchors是相对各个feature map

（每个feature map的感受野不同 检测的目标大小也不同 适合的anchor大小也不同）所以必须要顺序一致 否则效果会很不好。

这个函数一般用于check_anchors最后阶段。


```python
def check_anchor_order(m):
    """用在check_anchors最后 确定anchors和stride的顺序是一致的
    Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    :params m: model中的最后一层 Detect层
    """
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    # 计算anchor的面积 anchor area [9]
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
     # 计算最大anchor与最小anchor面积差
    da = a[-1] - a[0]  # delta a
    # 计算最大stride与最小stride差
    # m.stride: model strides 
    # https://github.com/Oneflow-Inc/one-yolov5/blob/bf8c66e011fcf5b8885068074ffc6b56c113a20c/models/yolo.py#L144-L152
    ds = m.stride[-1] - m.stride[0]  # delta s
    # flow.sign(x):当x大于/小于0时，返回1/-1
    # 如果这里anchor与stride顺序不一致，则重新调整顺序
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
```
## 3. kmean_anchors
&emsp;这个函数才是这个这个文件的核心函数，功能：使用K-means + 遗传算法 算出更符合当前数据集的anchors。

&emsp;这里不仅仅使用了k-means聚类，还使用了Genetic Algorithm遗传算法，在k-means聚类的结果上进行mutation变异。接下来简单介绍下代码流程：

1. 载入数据集，得到数据集中所有数据的wh
2. 将每张图片中wh的最大值等比例缩放到指定大小img_size，较小边也相应缩放
3. 将bboxes从相对坐标改成绝对坐标（乘以缩放后的wh） 
4. 筛选bboxes，保留wh都大于等于两个像素的bboxes
5. 使用k-means聚类得到n个anchors（调用k-means包 涉及一个白化操作）
6. 使用遗传算法随机对anchors的wh进行变异，如果变异后效果变得更好（使用anchor_fitness方法计算得到的fitness（适应度）进行评估）就将变异后的结果赋值给anchors，如果变异后效果变差就跳过，默认变异1000次
> 不知道什么是遗传算法，可以看看这两个b站视频：[传算法超细致+透彻理解](https://www.bilibili.com/video/BV1zp4y1U7Ti?from=search&seid=3206758960880461786)
和[霹雳吧啦Wz](https://www.bilibili.com/video/BV1Tv411T7qa?spm_id_from=333.851.dynamic.content.click)



```python
def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """在check_anchors中调用
    使用K-means + 遗传算法 算出更符合当前数据集的anchors
    Creates kmeans-evolved anchors from training dataset
    :params path: 数据集的路径/数据集本身
    :params n: anchor框的个数
    :params img_size: 数据集图片约定的大小
    :params thr: 阈值 由hyp['anchor_t']参数控制
    :params gen: 遗传算法进化迭代的次数(突变 + 选择)
    :params verbose: 是否打印所有的进化(成功的)结果 默认传入是Fasle的 只打印最佳的进化结果即可
    :return k: k-means + 遗传算法进化 后的anchors
    """
    from scipy.cluster.vq import kmeans


    # 注意一下下面的thr不是传入的thr，而是1/thr, 所以在计算指标这方面还是和check_anchor一样
    thr = 1. / thr  # 0.25
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # compute metrics
        """用于print_results函数和anchor_fitness函数
        计算ratio metric: 整个数据集的gt框与anchor对应宽比和高比即:gt_w/k_w,gt_h/k_h + x + best_x  用于后续计算bpr+aat
        注意我们这里选择的metric是gt框与anchor对应宽比和高比 而不是常用的iou 这点也与nms的筛选条件对应 是yolov5中使用的新方法
        :params k: anchor框
        :params wh: 整个数据集的wh [N, 2]
        :return x: [N, 9] N个gt框与所有anchor框的宽比或高比(两者之中较小者)
        :return x.max(1)[0]: [N] N个gt框与所有anchor框中的最大宽比或高比(两者之中较小者)
        """
        # [N, 1, 2] / [1, 9, 2] = [N, 9, 2]  N个gt_wh和9个anchor的k_wh宽比和高比
        # 两者的重合程度越高 就越趋近于1 远离1(<1 或 >1)重合程度都越低
        r = wh[:, None] / k[None]
        # r=gt_height/anchor_height  gt_width / anchor_width  有可能大于1，也可能小于等于1
        # flow.min(r, 1. / r): [N, 9, 2] 将所有的宽比和高比统一到<=1
        # .min(2): value=[N, 9] 选出每个gt个和anchor的宽比和高比最小的值   index: [N, 9] 这个最小值是宽比(0)还是高比(1)
        # [0] 返回value [N, 9] 每个gt个和anchor的宽比和高比最小的值 就是所有gt与anchor重合程度最低的
        x = flow.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, flow.tensor(k))  # IoU metric
        # x.max(1)[0]: [N] 返回每个gt和所有anchor(9个)中宽比/高比最大的值
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):   # mutation fitness
        """用于kmean_anchors函数
        适应度计算 优胜劣汰 用于遗传算法中衡量突变是否有效的标注 如果有效就进行选择操作 没效就继续下一轮的突变
        :params k: [9, 2] k-means生成的9个anchors     wh: [N, 2]: 数据集的所有gt框的宽高
        :return (best * (best > thr).float()).mean()=适应度计算公式 [1] 注意和bpr有区别 这里是自定义的一种适应度公式
                返回的是输入此时anchor k 对应的适应度
        """
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        """用于kmean_anchors函数中打印k-means计算相关信息
        计算bpr、aat=>打印信息: 阈值+bpr+aat  anchor个数+图片大小+metric_all+best_mean+past_mean+Kmeans聚类出来的anchor框(四舍五入)
        :params k: k-means得到的anchor k
        :return k: input
        """
        # 将k-means得到的anchor k按面积从小到大啊排序
        k = k[np.argsort(k.prod(1))]
        # x: [N, 9] N个gt框与所有anchor框的宽比或高比(两者之中较小者)
        # best: [N] N个gt框与所有anchor框中的最大 宽比或高比(两者之中较小者)
        x, best = metric(k, wh0)
        # (best > thr).float(): True=>1.  False->0.  .mean(): 求均值
        # bpr(best possible recall): 最多能被召回(通过thr)的gt框数量 / 所有gt框数量  [1] 0.96223  小于0.98 才会用k-means计算anchor
        # aat(anchors above threshold): [1] 3.54360 每个target平均有多少个anchors
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        f = anchor_fitness(k)
        # print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        # print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
        #       f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        print(f"aat: {aat:.5f}, fitness: {f:.5f}, best possible recall: {bpr:.5f}")
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg

        return k


    # 载入数据集
    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # 得到数据集中所有数据的wh
    # 将数据集图片的最长边缩放到img_size, 较小边相应缩放
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 将原本数据集中gt boxes归一化的wh缩放到shapes尺度
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])

    # 统计gt boxes中宽或者高小于3个像素的个数, 目标太小 发出警告
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')

    # 筛选出label大于2个像素的框拿来聚类,[...]内的相当于一个筛选器,为True的留下
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans聚类方法: 使用欧式距离来进行聚类
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} gt boxes...')
    # 计算宽和高的标准差->[w_std,h_std]
    s = wh.std(0)  # sigmas for whitening
    # 开始聚类,仍然是聚成n类,返回聚类后的anchors k(这个anchor k是白化后数据的anchor框)
    # 另外还要注意的是这里的kmeans使用欧式距离来计算的
    # 运行k-means的次数为30次  obs: 传入的数据必须先白化处理 'whiten operation'
    # 白化处理: 新数据的标准差=1 降低数据之间的相关度，不同数据所蕴含的信息之间的重复性就会降低，网络的训练效率就会提高
    # 白化操作博客: https://blog.csdn.net/weixin_37872766/article/details/102957235
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s  # k*s 得到原来数据(白化前)的anchor框

    wh = flow.tensor(wh, dtype=flow.float32)  # filtered wh
    wh0 = flow.tensor(wh0, dtype=flow.float32)  # unfiltered wh0

    # 输出新算的anchors k 相关的信息
    k = print_results(k)

    # Plot wh
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0], 400)
    # ax[1].hist(wh[wh[:, 1]<100, 1], 400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve 类似遗传/进化算法  变异操作
    npr = np.random   # 随机工具
    # f: fitness 0.62690
    # sh: (9,2)
    # mp: 突变比例mutation prob=0.9   s: sigma=0.1
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    # 根据聚类出来的n个点采用遗传算法生成新的anchor
    for _ in pbar:
        # 重复1000次突变+选择 选择出1000次突变里的最佳anchor k和最佳适应度f
        v = np.ones(sh)  # v [9, 2] 全是1
        while (v == 1).all():
            # 产生变异规则 mutate until a change occurs (prevent duplicates)
            # npr.random(sh) < mp: 让v以90%的比例进行变异  选到变异的就为1  没有选到变异的就为0
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        # 变异(改变这一时刻之前的最佳适应度对应的anchor k)
        kg = (k.copy() * v).clip(min=2.0)
        # 计算变异后的anchor kg的适应度
        fg = anchor_fitness(kg)
        # 如果变异后的anchor kg的适应度>最佳适应度k 就进行选择操作
        if fg > f:
            # 选择变异后的anchor kg为最佳的anchor k 变异后的适应度fg为最佳适应度f
            f, k = fg, kg.copy()

            # 打印信息
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)
    return print_results(k)
```

## 4. check_anchors

这个函数是通过计算bpr确定是否需要改变anchors 需要就调用k-means重新计算anchors。


```python
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary  
    """用于train.py中
    通过bpr确定是否需要改变anchors 需要就调用k-means重新计算anchors
    Check anchor fit to data, recompute if necessary
    :params dataset: 自定义数据集LoadImagesAndLabels返回的数据集
    :params model: 初始化的模型
    :params thr: 超参中得到  界定anchor与label匹配程度的阈值
    :params imgsz: 图片尺寸 默认640
    """
    # 从model中取出最后一层(Detect)
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]  # Detect()
    # dataset.shapes.max(1, keepdims=True) = 每张图片的较长边
    # shapes: 将数据集图片的最长边缩放到img_size, 较小边相应缩放 得到新的所有数据集图片的宽高 [N, 2]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 产生随机数scale [img_size, 1]
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # [6301, 2]  所有target(6301个)的wh   基于原图大小    shapes * scale: 随机化尺度变化
    wh = flow.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """用在check_anchors函数中  compute metric
        根据数据集的所有图片的wh和当前所有anchors k计算 bpr(best possible recall) 和 aat(anchors above threshold)
        :params k: anchors [9, 2]  wh: [N, 2]
        :return bpr: best possible recall 最多能被召回(通过thr)的gt框数量 / 所有gt框数量小于0.98 才会用k-means计算anchor
        :return aat: anchors above threshold 每个target平均有多少个anchors
        """
        # None添加维度  所有target(gt)的wh wh[:, None] [6301, 2]->[6301, 1, 2]
        #             所有anchor的wh k[None] [9, 2]->[1, 9, 2]
        # r: target的高h宽w与anchor的高h_a宽w_a的比值，即h/h_a, w/w_a  [6301, 9, 2]  有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        # x 高宽比和宽高比的最小值 无论r大于1，还是小于等于1最后统一结果都要小于1   [6301, 9]
        x = flow.min(r, 1 / r).min(2)[0]  # ratio metric
        # best [6301] 为每个gt框选择匹配所有anchors宽高比例值最好的那一个比值
        best = x.max(1)[0]  # best_x
        # aat(anchors above threshold)  每个target平均有多少个anchors
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        # bpr(best possible recall) = 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    # anchors: [N,2]  所有anchors的宽高   基于缩放后的图片大小(较长边为640 较小边相应缩放)
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    # 考虑这9类anchor的宽高和gt框的宽高之间的差距, 如果bpr<0.98(说明当前anchor不能很好的匹配数据集gt框)就会根据k-means算法重新聚类新的anchor
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
        na = m.anchors.numel() // 2  # number of anchors
        try:
            # 如果bpr<0.98(最大为1 越大越好) 使用k-means + 遗传进化算法选择出与数据集更匹配的anchors框  [9, 2]
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f"{PREFIX}ERROR: {e}")
        # 计算新的anchors的new_bpr
        new_bpr = metric(anchors)[0]
        # 比较 k-means + 遗传进化算法进化后的anchors的new_bpr和原始anchors的bpr
        # 注意: 这里并不一定进化后的bpr必大于原始anchors的bpr, 因为两者的衡量标注是不一样的  进化算法的衡量标准是适应度 而这里比的是bpr
        if new_bpr > bpr:  # replace anchors
            anchors = flow.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            # 替换m的anchor_grid                      [9, 2] -> [3, 1, 3, 1, 1, 2]
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            # 检查anchor顺序和stride顺序是否一致 不一致就调整
            # 因为我们的m.anchors是相对各个 feature map 所以必须要顺序一致 否则效果会很不好
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
        else:
            s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
        LOGGER.info(s)
```

这个函数会在[train.py中调用：](https://github.com/Oneflow-Inc/one-yolov5/blob/640ac163ee26a8b13bb2e94f348fb3752a250886/train.py#L252-L253)

![image](https://user-images.githubusercontent.com/109639975/199909323-103aaf2f-cdcd-4601-9faf-4618d08d3558.png)



## 总结
k-means是非常经典且有效的聚类方法，通过计算样本之间的距离（相似程度）将较近的样本聚为同一类别（簇）。

## Reference
- [YOLO9000:Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
- 【YOLOV5-5.x 源码解读】[autoanchor.py] https://blog.csdn.net/qq_38253797/article/details/119713706
- CSDN 霹雳吧啦Wz : [使用k-means聚类anchors](https://blog.csdn.net/qq_37541097/article/details/119647026?spm=1001.2014.3001.5501)
- Bilibili 霹雳吧啦Wz : [如何使用k-means聚类得到anchors以及需要注意的坑.](https://www.bilibili.com/video/BV1Tv411T7qa)
- CSDN 恩泽君 : [YOLOV3中k-means聚类获得anchor boxes过程详解.](https://github.com/Laughing-q/yolov5_annotations/blob/master/utils/autoanchor.py)
- Github 恩泽君: [Laughing-q/yolov5_annotations.](https://github.com/Laughing-q/yolov5_annotations/blob/master/utils/autoanchor.py)
- CSDN 昌山小屋: 【玩转yolov5】[请看代码之自动anchor计算.](https://blog.csdn.net/ChuiGeDaQiQiu/article/details/113487612?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162899414216780265433994%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162899414216780265433994&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-113487612.pc_search_result_control_group&utm_term=best+possible+recall&spm=1018.2226.3001.4187)
- CSDN TheOldManAndTheSea: [目标检测 YOLOv5 anchor设置](https://flyfish.blog.csdn.net/article/details/117594265)
- Bilibili 我家公子Q: [遗传算法超细致+透彻理解](https://www.bilibili.com/video/BV1zp4y1U7Ti?from=search&seid=3206758960880461786)

