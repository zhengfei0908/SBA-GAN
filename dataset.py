from io import BytesIO

import lmdb
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
from pytorch_pretrained_bert import BertTokenizer


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=4, max_length=18):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.max_length = max_length
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img_key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(img_key)
            for k in range(3):
                i = random.randint(0, 5)
                txt_key = f'txt-{i}-{str(index).zfill(5)}'.encode('utf-8')
                txt_bytes = txn.get(txt_key)
                if txt_bytes is not None:
                    break
            else:
                txt_bytes = '[CLS] [SEP]'.encode('utf-8')

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        
        ####Todo: text preprocessing, stop words####
        txt = txt_bytes.decode('utf-8')
        txt = '[CLS] ' + txt.replace("\ufffd\ufffd", " ") + ' [SEP]'
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(txt))
        fixed_tokens = torch.zeros(self.max_length, dtype=torch.int64)
        fixed_tokens[:min(len(tokens), self.max_length)] = torch.LongTensor(tokens[:min(len(tokens), self.max_length)])
        
        return img, fixed_tokens