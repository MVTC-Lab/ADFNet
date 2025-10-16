# ADFNet
## ADFNet——基于自适应双模式频域增强的时间序列预测网络。


1）您可以在exp文件夹下见到各数据集的运行脚本（参数文件），为了尽量保证可复现性，您可以将其中的参数复制到run.py文件的相应位置。

2）models文件夹下是ADFNet（SCINet.py）的网络模型。

3）需要根据您的文件路径修改run.py和experiments文件夹下的exp.py的相关路径（这是相当简单的事）。

4）torch版本为2.0.0，python版本为3.8，需要安装的依赖可见requirements.txt，安装包的命令如下（将packname替换为所需安装的包即可）：{pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple packname}