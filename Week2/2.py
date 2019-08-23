import numpy as np
import imageio


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d


filename = 'cifar-10-batches-py'

meta = unpickle(filename + '/batches.meta')
label_name = meta[b'label_names']
print(label_name)

for i in range(1, 6):
    content = unpickle(filename + '/data_batch_' + str(i))
    print('load data...')
    print(content.keys())
    print('transferring data_batch' + str(i))
    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = 'train/' + label_name[content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(j) + '.jpg'
        imageio.imwrite(img_name, img)
