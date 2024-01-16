# Assocation reasoning

## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练

  

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: 网络模块
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**
* 训练过程中保存的`results.txt`是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
* 在使用预测脚本时，要将`train_weights`设置为你自己生成的权重路径。
