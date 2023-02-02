## 前言

>🎉代码仓库地址：<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank">https://github.com/Oneflow-Inc/one-yolov5</a>
欢迎star [one-yolov5项目](https://github.com/Oneflow-Inc/one-yolov5) 获取<a href="https://github.com/Oneflow-Inc/one-yolov5/tags" target="blank" >最新的动态。</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5/issues/new"  target="blank"  >如果您有问题，欢迎在仓库给我们提出宝贵的意见。🌟🌟🌟</a>
<a href="https://github.com/Oneflow-Inc/one-yolov5" target="blank" >
如果对您有帮助，欢迎来给我Star呀😊~  </a>


源码解读： [callbacks.py](https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/callbacks.py)

> 这个文件是yolov5的Callback utils

## 钩子
![Hook](https://foruda.gitee.com/images/1675324604046868984/6e5bbb58_10213136.png)


hook(钩子)是一个编程机制，与语言无关，通常用于在不修改原始代码的情况下，捕获或替换程序的一些函数或API调用。

个人观点：钩子是指将代码插入到其他代码的执行流程中的技术，从而实现在执行原有代码之前或之后执行额外代码的目的。

## 回调函数
来源网络的例子，有一家旅馆提供叫醒服务，但是要求旅客自己决定叫醒的方法。可以是打客房电话，也可以是派服务员去敲门，睡得死怕耽误事的，还可以要求往自己头上浇盆水。这里，“叫醒”这个行为是旅馆提供的，相当于库函数，但是叫醒的方式是由旅客决定并告诉旅馆的，也就是回调函数。而旅客告诉旅馆怎么叫醒自己的动作，也就是把回调函数传入库函数的动作，称为登记回调函数（to register a callback function）。如下图所示（图片来源：维基百科）：

![callback](https://foruda.gitee.com/images/1675328186654129000/73d1be7b_10213136.png)


上图可以看到，回调函数通常和应用处于同一抽象层（因为传入什么样的回调函数是在应用级别决定的）。而回调就成了一个高层调用底层，底层再回过头来调用高层的过程。

简单来说：
- 一般函数：function a(int a, String b)，接收的参数是一般类型。
- 特殊函数：function b(function c)，接收的参数是一个函数，c这个函数就叫**回调函数**。

个人观点：回调函数是指在代码中被调用的一个函数，它会对其他代码的执行造成影响，并在适当的时间进行回调。

**总之，钩子和回调函数是实现代码间通信和协作的不同技术，它们都可以用于实现代码级别的自定义行为，只是函数的触发时机有差异。**


## hook实现例子
> hook函数是程序中预定义好的函数，这个函数处于原有程序流程当中（暴露一个钩子出来）。
我们需要再在有流程中钩子定义的函数块中实现某个具体的细节，需要把我们的实现，挂接或者注册（register）到钩子里，使得hook函数对目标可用。

hook函数最常使用在某种流程处理当中。这个流程往往有很多步骤。hook函数常常挂载在这些步骤中，为增加额外的一些操作，提供灵活性。

下面举一个简单的例子，这个例子的目的是实现一个通过钩子调用函数判断字符串是否是"good"


```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""
class Callbacks:
    """ "
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks
        self._callbacks = {
            "on_pretrain_routine_start": [],
        }
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to 要向其注册操作的回调钩子名称
            name: The name of the action for later reference 动作的名称，供以后参考
            callback: The callback to fire 对fire的回调
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """ "
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            kwargs: Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger["callback"](*args, **kwargs)
```


```python
def on_pretrain_routine_start(good:str):
    if good == "good":
        print("is good!")
    else :
        print("is bad!")
```


```python
# 初始化 Callbacks 对象
callbacks=Callbacks()
# 要向其注册操作的回调钩子名称
callbacks.register_action(hook = "on_pretrain_routine_start",name = "ss" , callback=on_pretrain_routine_start)
# 调用hook
callbacks.run("on_pretrain_routine_start","good")
# 打印hook信息
callbacks.get_registered_actions("on_pretrain_routine_start")
```

    is good





    [{'name': 'ss',
      'callback': <function __main__.on_pretrain_routine_start(good: str)>}]




##   yolov5项目中

在yolov5训练流程中，hook函数体现在
一个训练过程(不包括数据准备)，会轮询多次训练集，每次称为一个epoch，每个epoch又分为多个batch来训练。
流程先后拆解成:
- 开始训练
- 训练一个epoch前
- 训练一个batch前
- 训练一个batch后
- 训练一个epoch后。
- 评估验证集
- 结束训练

这些步骤是穿插在训练一个batch数据的过程中，这些可以理解成是钩子函数，我们可能需要在这些钩子函数中实现一些定制化的东西，比如在训练一个epoch后我们要保存下训练的损失。


```python
# 在train.py中hook注册操作代码
# Register actions
for k in methods(loggers):
    callbacks.register_action(k, callback=getattr(loggers, k))
```


```python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """ "
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks 
        # 定义些回调函数，函数实现在utils/loggers/__init__.py 
        # github链接: https://github.com/Oneflow-Inc/one-yolov5/blob/main/utils/loggers/__init__.py
        self._callbacks = {
            "on_pretrain_routine_start": [],
            # https://github.com/Oneflow-Inc/one-yolov5/blob/88864544cd9fa9ddcbe35a28a0bcf2c674daeb97/utils/loggers/__init__.py#L118
            "on_pretrain_routine_end": [], 
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],  # fit = train + val
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """ "
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            kwargs: Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger["callback"](*args, **kwargs)
```
