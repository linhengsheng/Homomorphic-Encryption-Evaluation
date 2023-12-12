[GetStart](https://github.com/OpenMined/TenSEAL/blob/main/tutorials%2FTutorial%200%20-%20Getting%20Started.ipynb)   
 CKKS 方案中，尺度（scale）是一个重要的概念。尺度用于确定加密数据的精度和范围。   
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193) 
poly_modulus_degree是多项式模数的度，即多项式的最高次数 
plain_modulus是明文模数 
较大的poly_modulus_degree意味着更高的安全性和计算复杂性 
plain_modulus限定了明文的范围 
 
SQLite数据库 
ts 序列化: serialize, 反序列化: tensor_from(context, enc_ser), context_from(key_ser)
 
即使是同一个上下文和同一个明文，每次加密后的密文分布无法区分 
# utils.py主要函数介绍
### cut_image(image_path, patch_size=64) 
    将输入图像裁剪成一定大小的patch，保存到目录：images/origin_dir中
### reconstruct(directory) 
    将image_dir中的patches重组
### del_file(path) 
    删除文件
### del_dir(directory) 
    删除目录下的所有文件
### cosine_s(cipher1, cipher2) -> float
    计算两组tenseal密文向量的明文相似度并返回
### cosine_s_sub(vector1, vector2)
    计算两个tenseal密文向量的明文相似度并返回
### delete_table(path, table) 
    删除数据库path中的某个表
### init_db(path) 
    初始化数据库
### insert_cipher(path, image_name, ciphers) 
    往数据中插入加密后的图像，ciphers为一个列表，元素类型为tenseal库的密文向量：tenseal.tensors.vector
### get_cipher(path, image_name) -> tuple 
    从数据库中获取image_name的所有密文
### clear_db(path) 
    清除数据库中的数据（不删除表结构）
### optimize_db(path) 
    优化数据库数据，把数据库中重复的数据删除，保证唯一性
### enc_image(context_type, context, image_path) 
    加密图像，返回图像的像素密文向量
### get_images(image_dir) -> list
    从image_dir中获取所有图像文件的路径
### search_cipher(db_path, enc_type, encrypter, in_ciphers)
    根据输入的密文向量搜索图像，返回图像的名称和密文以及相似度(相似度阈值大于0.9999)
### equal_img(image1, image2) 
    明文下验证两张图像是否相等
### cipher2img(context_type, decrypter, ciphers, shape) 
    将的密文向量转化成图像
### evaluate(vector, test_num=100) 
    测试同态加法和dot运算以及mul运算的用时
### class KeyManage:
    管理tenseal上下文的公私钥
