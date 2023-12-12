from cv2 import imread
import tenseal as ts
import time
import utils

ckks_enc_dir = './images/ckks_enc_dir'
bfv_enc_dir = './images/bfv_enc_dir'
image_dir = './images/origin_dir'
bfv_db = './bfv.db'
ckks_db = './ckks.db'
test_path = './images/origin_dir/test_67.png'
test_name = test_path.split('/')[-1].split('.')[0]
shape = imread(test_path).shape

with open('context/bfv_publicContext.pkl', 'rb') as f:
    bfv_encryptor = ts.context_from(f.read())
with open('context/bfv_secretContext.pkl', 'rb') as f:
    bfv_decrypter = ts.context_from(f.read())
with open('context/ckks_publicContext.pkl', 'rb') as f:
    ckks_encryptor = ts.context_from(f.read())
with open('context/ckks_secretContext.pkl', 'rb') as f:
    ckks_decrypter = ts.context_from(f.read())

if __name__ == '__main__':
    print('clear data')
    utils.delete_table(ckks_db, 'ciphers')
    utils.del_dir(bfv_enc_dir)
    utils.del_dir(ckks_enc_dir)
    # print('cut image')
    # utils.cut_image('./images/miku.png')
    # print('reconstruct image')
    # utils.reconstruct(image_dir)
    # print('image cut success or not: ', utils.equal_img('./images/miku.png', 'images/reconstruct_img.png'))
    # utils.init_db(bfv_db)
    utils.init_db(ckks_db)
    # optimize_db(db_path)
    print('KeyGen')
    time_0 = time.time()
    keyManage = utils.KeyManage()
    with open('./context/bfv_secretContext.pkl', 'wb') as f:
        f.write(keyManage.get_bfv_secretKey())
    with open('./context/bfv_publicContext.pkl', 'wb') as f:
        f.write(keyManage.get_bfv_publicKey())
    with open('./context/ckks_secretContext.pkl', 'wb') as f:
        f.write(keyManage.get_ckks_secretKey())
    with open('./context/ckks_publicContext.pkl', 'wb') as f:
        f.write(keyManage.get_ckks_publicKey())
    time_1 = time.time()
    print(f'time of keyGen: {(time_1-time_0)*1000}ms')
    print('Done')


