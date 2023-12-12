from cv2 import imread, imwrite
import numpy as np
import os
import random
import tenseal as ts
import time
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3


def is_prime(n, k=10):
    """Miller-Rabin素数测试"""
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # 将 n - 1 表示为 2^r * d 的形式
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # 进行 Miller-Rabin 测试
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_large_prime(bits):
    """生成几十比特位的大素数"""
    while True:
        candidate = random.getrandbits(bits)
        if candidate % 2 == 0:
            candidate += 1  # 确保是奇数
        if is_prime(candidate):
            return candidate


def cut_image(image_path, patch_size=64):
    img = imread(image_path)
    h, w, c = img.shape
    patch_size = 64
    for x in range(w // patch_size):
        for y in range(h // patch_size):
            sub = img[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :]
            imwrite(f'./images/origin_dir/test_{x * 10 + y}.png', sub)


def reconstruct(directory):
    paths = os.listdir(directory)
    for i, path in enumerate(paths):
        if path.split('.')[-1] not in ['jpeg', 'jpg', 'png', 'gif', 'bmp']:
            del paths[i]
    scale = np.sqrt(len(paths)).astype(int)
    sub = imread('/'.join([directory, paths[0]]))
    h, w, c = sub.shape
    ret_img = np.empty([h * scale, w * scale, c])
    for x in range(scale):
        for y in range(scale):
            sub = imread('/'.join([directory, paths[x * scale + y]]))
            ret_img[x * h:(x + 1) * h, y * w:(y + 1) * w, :] = sub
    imwrite('./images/reconstruct_img.png', ret_img)


def del_file(path):
    try:
        # 删除文件
        os.remove(path)
    except FileNotFoundError:
        print(f"File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def del_dir(directory):
    list_paths = os.listdir(directory)
    for path in list_paths:
        del_file('/'.join([directory, path]))


def cosine_s_sub(vector1, vector2):
    m = vector1.dot(vector2)
    d1 = vector1.dot(vector1).decrypt()
    d2 = vector2.dot(vector2).decrypt()
    m = np.array(m.decrypt())
    d = np.array(d1) * np.array(d2)
    if d > 0:
        return m / np.sqrt(d)
    else:
        print(m)
        print(d)
        print('out of range')
        return None


def cosine_s(cipher1, cipher2) -> float:
    scores = []
    for b1, b2 in zip(cipher1, cipher2):
        score = cosine_s_sub(b1, b2)
        if score < 0.9999:
            return 0
        scores.append(score)
    score = np.array(scores).mean()
    return score


def delete_table(path, table):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE {table}')
    # 提交更改
    conn.commit()
    conn.close()


def init_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS ciphers (
            image_id INTEGER PRIMARY KEY,
            image_name TEXT NOT NULL,
            cipher_id INTEGER,
            cipher BLOB,
            UNIQUE (image_name, cipher_id)
        )'''
    )
    # 提交更改
    conn.commit()
    conn.close()


# 插入密文
def insert_cipher(path, image_name, ciphers):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    data = [(image_name, index, cipher.serialize()) for index, cipher in enumerate(ciphers)]
    cursor.executemany(f'''INSERT INTO ciphers (image_name, cipher_id, cipher) VALUES (?, ?, ?)''',
                       data)
    conn.commit()
    conn.close()


# 从表中读取序列化的image密文数据
def get_cipher(path, image_name) -> list:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(f'SELECT cipher FROM ciphers WHERE image_name = ?', (image_name,))
    result = cursor.fetchall()
    ciphers = [c[0] for c in result]
    conn.close()
    return ciphers


# 清除表中数据
def clear_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM ciphers')
    conn.commit()
    conn.close()


def optimize_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    save = '''SELECT image_name, cipher_id, cipher
                    FROM ciphers
                    GROUP BY image_name, cipher_id
                    HAVING COUNT(*) > 1
                '''
    cursor.execute(save)
    ciphers = cursor.fetchall()
    conn.commit()
    delete = '''DELETE FROM ciphers
            WHERE (image_name, cipher_id) IN
            (SELECT image_name, cipher_id
                FROM ciphers
                GROUP BY image_name, cipher_id
                HAVING COUNT(*) > 1);
            '''
    cursor.execute(delete)
    conn.commit()
    cursor.executemany(f'''INSERT INTO ciphers (image_name, cipher_id, cipher) VALUES (?, ?, ?)''',
                       ciphers)
    conn.commit()
    conn.close()


def enc_image(context_type, context, image_path):
    img_vector = imread(image_path)
    h, w, c = img_vector.shape
    img_vectors = img_vector.reshape([c, h*w])/255
    if context_type == ts.SCHEME_TYPE.BFV:
        enc_vectors = []
        for img_vector in img_vectors:
            enc_vector = ts.bfv_vector(context, img_vector)
            enc_vectors.append(enc_vector)
    elif context_type == ts.SCHEME_TYPE.CKKS:
        enc_vectors = []
        for img_vector in img_vectors:
            enc_vector = ts.ckks_vector(context, img_vector)
            enc_vectors.append(enc_vector)
    else:
        enc_vectors = None
    return enc_vectors


def get_images(image_dir) -> list:
    paths = os.listdir(image_dir)
    for i, path in enumerate(paths):
        if path.split('.')[-1] not in ['jpeg', 'jpg', 'png', 'gif', 'bmp']:
            del paths[i]
    for i, path in enumerate(paths):
        paths[i] = '/'.join([image_dir, path])
    return paths


# 返回图像名称，图像密文，以及相似度
def search_cipher(db_path, enc_type, encrypter, in_ciphers):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT image_name FROM ciphers where cipher_id = 0''')
    ret = cursor.fetchall()
    # print(len(ret))
    if ret is None:
        print('no images in database')
        conn.close()
        return None, None, None
    if enc_type == ts.SCHEME_TYPE.CKKS:
        for image in ret:
            image_name = image[0]
            # print(f'image name: {image_name}')
            ciphers = get_cipher(db_path, image_name)
            ciphers = [ts.ckks_vector_from(encrypter, cipher) for cipher in ciphers]
            # print('count cosine_similarity')
            start_time = time.time()
            score = cosine_s(in_ciphers, ciphers)
            end_time = time.time()
            count_time = (end_time - start_time)*1000
            if score > 0.999:
                return image_name, ciphers, score, count_time
    print('no such image')
    return None, None, None


# 判断两张图片是否相同
def equal_img(image1, image2):
    img1 = imread(image1)
    img2 = imread(image2)
    return np.array_equal(img1, img2)


def cipher2img(context_type, ciphers, shape):
    image = []
    if context_type == ts.SCHEME_TYPE.BFV:
        for cipher in ciphers:
            image.append(cipher.decrypt())
    elif context_type == ts.SCHEME_TYPE.CKKS:
        for cipher in ciphers:
            image.append(cipher.decrypt())
    else:
        return None
    dec = (np.array(image)*255).reshape(shape)
    return dec


def evaluate(vector, test_num=100):
    add_times = []
    mul_times = []
    dot_times = []
    for i in range(test_num):
        add_start_time = time.time()
        temp = vector.add(vector)
        add_end_time = time.time()
        temp = vector.dot(vector)
        dot_end_time = time.time()
        temp = vector.mul(vector)
        mul_end_time = time.time()
        add_time = (add_end_time - add_start_time)*1000
        mul_time = (dot_end_time - add_end_time)*1000
        dot_time = (mul_end_time - dot_end_time)*1000
        add_times.append(add_time)
        mul_times.append(mul_time)
        dot_times.append(dot_time)
    avl_add_time = np.array(add_times).mean()
    avl_mul_time = np.array(mul_times).mean()
    avl_dot_time = np.array(dot_times).mean()
    return avl_add_time, avl_dot_time, avl_mul_time


class KeyManage:
    def __init__(self):
        # coeff_mod_bit_sizes指定每个系数模数的大小
        self.ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                                       coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.ckks_context.global_scale = pow(2, 40)
        self.ckks_context.generate_galois_keys()
        self.ckks_context.generate_relin_keys()
        self.bfv_context = ts.context(ts.SCHEME_TYPE.BFV, 8192, plain_modulus=1032193)
        self.bfv_context.generate_galois_keys()

        # 分发 CKKS 公钥
    def get_ckks_publicKey(self) -> bytes:
        return self.ckks_context.serialize(save_public_key=True,
                                           save_secret_key=False,
                                           save_relin_keys=True,
                                           save_galois_keys=True)

        # 分发 CKKS 所有密钥
    def get_ckks_secretKey(self) -> bytes:
        return self.ckks_context.serialize(save_public_key=True,
                                           save_secret_key=True,
                                           save_relin_keys=True,
                                           save_galois_keys=True)

        # 分发 BFV 公钥
    def get_bfv_publicKey(self) -> bytes:
        return self.bfv_context.serialize(save_public_key=True,
                                          save_secret_key=False,
                                          save_relin_keys=True,
                                          save_galois_keys=True)

        # 分发 BFV 所有密钥
    def get_bfv_secretKey(self) -> bytes:
        return self.bfv_context.serialize(save_public_key=True,
                                          save_secret_key=True,
                                          save_relin_keys=True,
                                          save_galois_keys=True)


