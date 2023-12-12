from setup import ckks_decrypter, test_path
from utils import evaluate
import tenseal as ts
from cv2 import imread
import numpy as np


image = imread(test_path)
h, w, c = image.shape
image = image.reshape(h, w*c)

ckks_add_times = []
ckks_mul_times = []
ckks_dot_times = []
for index, arr in enumerate(image):
    ckks_vector = ts.ckks_vector(ckks_decrypter, arr)
    add_time, dot_time, mul_time = evaluate(ckks_vector, test_num=20)
    ckks_add_times.append(add_time)
    ckks_mul_times.append(mul_time)
    ckks_dot_times.append(dot_time)
    print(f'CKKS     add_time: {add_time: .13f}ms    mul_time: {mul_time: .13f}ms    dot_time: {dot_time: .13f}ms')
ckks_avl_add_time = np.array(ckks_add_times).mean()
ckks_avl_mul_time = np.array(ckks_mul_times).mean()
ckks_avl_dot_time = np.array(ckks_dot_times).mean()

print(f'CKKS     avl_add_time: {ckks_avl_add_time}ms    avl_mul_time: {ckks_avl_mul_time}ms    '
      f'avl_dot_time: {ckks_avl_dot_time}ms')
