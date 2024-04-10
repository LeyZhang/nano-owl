# API使用方法
## demo的使用
进入examples文件夹，修改里面的config/config.yml配置文件，根据实际想要的效果修改配置文件中的参数，然后运行api.py文件即可。

运行命令如下：

```bash
cd examples
python3 api.py
```

## api的使用

在api.py文件中，我们可以看到如下代码：

```python
def detect_api(image_path, prompt)
```

这函数就是检测的api接口，用于检测一张图片中的prompt物体，返回检测到的物体标签、置信度、bbox的信息。

使用这个api之前，必须首先加载模型，加载模型的代码如下：

```python  
def load_model(model, engine_path)
```

这个函数用于加载模型，model是模型的名称，engine_path是模型的路径。

综上所述，使用api的步骤如下：

1. 加载模型
2. 对于每张图片，调用api接口

