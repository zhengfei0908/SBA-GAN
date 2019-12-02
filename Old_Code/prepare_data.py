import argparse
import os
from glob import glob
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size, Image.LANCZOS)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(4, 8, 16, 32, 64, 128, 256, 512), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(transaction, img_path, txt_path, n_worker, dataset, sizes=(4, 8, 16, 32, 64, 128, 256, 512)):
    resize_fn = partial(resize_worker, sizes=sizes)
    
    if dataset == 'coco':
        img_files = sorted(os.listdir(img_path))
        img_files = [(i, os.path.join(img_path, img)) for i, img in enumerate(img_files) if img.endswith('.jpg')]

        txt_files = sorted(os.listdir(txt_path))
        txt_files = [os.path.join(txt_path, txt) for txt in txt_files if txt.endswith('.txt')]
    else:
        imgset = datasets.ImageFolder(img_path)
        img_files = sorted(imgset.imgs, key=lambda x: x[0])
        img_files = [(i, img) for i, (img, label) in enumerate(img_files) if img.endswith('.jpg')]
        
        txt_files = []
        for sub in os.listdir(txt_path):
            path = os.path.join(txt_path, sub)
            for txt in os.listdir(path):
                if txt.endswith('.txt'):
                    txt_files.extend([os.path.join(path, txt)])
                
        txt_files = sorted(txt_files)
        
    assert len(img_files) == len(txt_files), "length doesn't match, please check the folders and suffic names"
    
    print('Total samples: %d\n' %len(img_files))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, img_files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                # print(key)
                transaction.put(key, img)
            with open(txt_files[i], 'r') as f:
                captions = f.read().split('\n')
                for k, cap in enumerate(captions):
                    if len(cap) == 0:
                        continue
                    key = f'txt-{k}-{str(i).zfill(5)}'.encode('utf-8')
                    cap = cap.encode('utf-8')
                    transaction.put(key, cap)
            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--txt_path', type=str)
    parser.add_argument('--dataset', default='birds', type=str, choices=['birds', 'coco'])

    args = parser.parse_args()

    
    with lmdb.open(args.out, map_size=512 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, args.img_path, args.txt_path, args.n_worker, args.dataset)
