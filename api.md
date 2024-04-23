# 特征提取脚本
## 介绍
脚本位于 'examples/feature_extract.py'，用于提取图片的特征。使用它需要修改 'config/feature_extract.yml' 中的参数，参数具体意义注释在该配置文件中。

值得注意的是输入文件夹应当直接包含被提取特征的图像，而不是包含子文件夹，而且由于输出的特征文件较大，建议不要在没有行人的背景图片上提取特征，只在包含行人的图片上提取特征。

## 使用方法
进入examples文件夹，修改里面的config/feature_extract.yml配置文件，根据实际想要的效果修改配置文件中的参数，然后运行feature_extract.py文件。

运行命令如下：
    
```bash
cd examples
python3 feature_extract.py
```

# API使用方法（包含clf头）
## demo的使用
进入examples文件夹，修改里面的config/config.yml配置文件，根据实际想要的效果修改配置文件中的参数，然后运行api.py文件即可。注意需要包含clf头的模型，放在 'examples/model' 文件夹下。

运行命令如下：

```bash
cd examples
python3 api.py
```

## api的使用

在api.py文件中，我们可以看到如下代码：

```python
def detect_api(image_path, prompt, config)
```

这函数就是检测的api接口，用于检测一张图片中的prompt物体，返回检测到的物体标签、置信度、bbox的信息。这里的config是从配置文件中传入的必须参数，这是与上个版本的api变化的部分，具体含义可以在api.py文件的示例中查看。

使用这个api之前，必须首先加载模型，加载模型的代码如下：

```python  
def load_model(model, engine_path)
```

这个函数用于加载模型，model是模型的名称，engine_path是模型的路径。

这个版本的api使用方法与上个版本并没有过多改变，只需要注意clf模型的正确加载，其他部分的接口是基本不变的。

