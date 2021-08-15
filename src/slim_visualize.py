# 模型和敏感度文件可视化
import paddlex as pdx
model = pdx.load_model('output/resnet50_vd_ssld/best_model')
pdx.slim.visualize(model, 'resnet50_vd_ssld.sensi.data', save_dir='./')