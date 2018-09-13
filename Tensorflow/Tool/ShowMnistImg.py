import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# 导入数据
mnist = input_data.read_data_sets("./Data/Mnist/", one_hot = True)

# 读取一幅图像
read_idx = 309
img_np = mnist.train.images[read_idx]
# reshape
img_np = img_np.reshape(28, 28)
img = Image.fromarray(img_np * 255)
# 不转换保存会报错
img= img.convert('RGB')
img.save("./Data/Mnist/mnist_%05d.jpg"%read_idx)

# 显示
plt.imshow(img)
plt.show()
