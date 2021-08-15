#分析模型参数信息
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model('output/resnet50_vd_ssld/best_model')

eval_dataset = pdx.datasets.ImageNet(
    data_dir='image',
    file_list='image/val_list.txt',
    label_list='image/labels.txt',
    transforms=model.eval_transforms)

pdx.slim.prune.analysis(
    model,
    dataset=eval_dataset,
    batch_size=16,
    save_file='resnet50_vd_ssld.sensi.data')