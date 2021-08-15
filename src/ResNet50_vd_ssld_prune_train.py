# 模型进行裁剪训练
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.cls import transforms
import paddlex as pdx

train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-imagenet
train_dataset = pdx.datasets.ImageNet(
    data_dir='image',
    file_list='image/train_list.txt',
    label_list='image/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='image',
    file_list='image/val_list.txt',
    label_list='image/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
model = pdx.cls.ResNet50_vd_ssld(num_classes=len(train_dataset.labels))
# model = pdx.cls.ResNet50_vd_ssld(num_classes=1000)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/classification.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=200,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.01,
    save_interval_epochs=10,
    pretrain_weights='output/resnet50_vd_ssld/best_model',
    save_dir='output_optimize/resnet50_vd_ssld',
    sensitivities_file='./resnet50_vd_ssld.sensi.data',
    eval_metric_loss=0.05,
    use_vdl=True)