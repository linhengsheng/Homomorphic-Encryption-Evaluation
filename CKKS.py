import numpy as np
from cv2 import imread, imwrite
import tenseal as ts
import time
from setup import image_dir, test_path, test_name, ckks_decrypter, ckks_encryptor, ckks_db, shape, ckks_enc_dir
from utils import get_images, get_cipher, search_cipher, insert_cipher, optimize_db, \
    delete_table, clear_db, init_db, enc_image, cipher2img

enc_times = []
insert_times = []
# 加密并存储
image_paths = get_images(image_dir)
for index, image_path in enumerate(image_paths):
    image_name = image_path.split('/')[-1].split('.')[0]
    time_0 = time.time()
    # 加密图像，返回密文向量
    enc_vectors = enc_image(ts.SCHEME_TYPE.CKKS, ckks_encryptor, image_path)
    time_1 = time.time()
    # 插入数据库
    insert_cipher(ckks_db, image_name, enc_vectors)
    time_2 = time.time()
    enc_time = (time_1 - time_0) * 1000
    insert_time = (time_2 - time_1) * 1000
    print(index, f'加密所用时间: {enc_time}ms  密文插入数据库所用时间: {insert_time}ms')
    enc_times.append(enc_time)
    insert_times.append(insert_time)

avl_enc_time = np.array(enc_times).mean()
avl_insert_time = np.array(insert_times).mean()
print(
    f"图片大小: {shape}    使用算法: CKKS    平均加密用时: {avl_enc_time}ms    密文插入数据库平均用时: {avl_insert_time}ms")

search_times = []
count_times = []
true_num = 0
for image_path in image_paths:
    # print(image_path)
    enc_vectors = enc_image(ts.SCHEME_TYPE.CKKS, ckks_decrypter, image_path)
    start_time = time.time()
    name, ciphers, score, count_time = search_cipher(ckks_db, ts.SCHEME_TYPE.CKKS, ckks_decrypter, enc_vectors)
    end_time = time.time()
    # print('search done')
    search_time = (end_time - start_time) * 1000
    search_times.append(search_time)
    count_times.append(count_time)
    if name is not None:
        image = cipher2img(ts.SCHEME_TYPE.CKKS, ciphers, shape)
        image = np.round(image)
        test_image = imread(image_path)
        is_same = np.array_equal(test_image, image)
        print('find image: ', name, is_same, f"密文检索用时: {search_time: .13f}ms      索引相似度: {score:.13f}"
                                             f"   相似度计算用时: {count_time}ms")
        if is_same:
            true_num += 1

avl_search_time = np.array(search_times).mean()
avl_count_time = np.array(count_times).mean()
print(f"密文检索平均用时: {avl_search_time: .13f}ms   密文还原成功率: {true_num / len(image_paths)}"
      f"   平均相似度计算用时: {avl_count_time}ms")
