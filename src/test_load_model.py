#加载训练保存的模型预测
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from collections.abc import Iterable
import json
model = pdx.load_model('output/resnet50_vd_ssld/best_model')
image_name='image/Dianthus_chinensis/shizhu_117.jpg'
img=mpimg.imread(image_name)
result = model.predict(image_name)
print("Predict Result: ", result)
plt.imshow(img)
plt.text(5,15,str(result[0]))
plt.show()