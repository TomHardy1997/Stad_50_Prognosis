# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os
# import pandas as pd
# from PIL import Image

# # 自定义Dataset类
# class TumorDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = pd.read_csv(df)
#         self.transform = transform
#         self.tissue = 'TUM'
#         self.transform = transform
#         # import ipdb;ipdb.set_trace()

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         patient_path = self.df['path'].iloc[idx]
#         images = []
#         image_path = os.path.join(patient_path, self.tissue)
#         for path in os.listdir(image_path):
#             try:
#                 path = os.path.join(image_path,path)
#                 image = Image.open(path).convert('RGB')
#                 image = self.transform(image)
#                 images.append(image)
#             except Exception as e:
#                 print(f'{path} has found error')
#             patient_image = torch.stack(images)
#         return patient_image, patient_path
    
# if __name__ == '__main__':
#     transforms = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])])
#     dataset = TumorDataset(df='patient_for_hospital.csv',transform=transforms)
#     import ipdb;ipdb.set_trace()


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os
# import pandas as pd
# from PIL import Image
# import cv2  # 添加了opencv库

# # # 自定义Dataset类
# class TumorDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = pd.read_csv(df)
#         self.transform = transform
#         self.tissue = 'TUM'

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         patient_path = self.df['path'].iloc[idx]
#         images = []
#         image_path = os.path.join(patient_path, self.tissue)
#         for path in os.listdir(image_path):
#             try:
#                 path = os.path.join(image_path, path)
#                 # 使用opencv加载图像，可能更快
#                 image = cv2.imread(path)
#                 # 将BGR转换为RGB
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 # 转换为PIL Image
#                 image = Image.fromarray(image)
#                 if self.transform:
#                     image = self.transform(image)
#                 images.append(image)
#             except Exception as e:
#                 print(f'{path} has found error')
#         patient_image = torch.stack(images)
#         return patient_image, patient_path

# if __name__ == '__main__':
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])])
    
#     dataset = TumorDataset(df='patient_for_hospital.csv', transform=transforms)
#     # 使用更多的worker来加速数据加载
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
#     # 保持其他代码不变
#     import ipdb; ipdb.set_trace()

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os
# import pandas as pd
# import cv2
# import glob
# from PIL import Image
# import numpy as np

# class TumorDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = pd.read_csv(df)
#         self.transform = transform
#         self.tissue = 'TUM'

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         patient_path = self.df['path'].iloc[idx]
#         images = []
#         image_path = os.path.join(patient_path, self.tissue)
#         for path in glob.glob(f'{image_path}/*.png'):
#             try:
#                 image = cv2.imread(path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(image)
#                 if self.transform:
#                     image = self.transform(image)
#                 images.append(image)
#             except Exception as e:
#                 print(f'Error in {path}: {e}')
#         patient_image = torch.stack(images)
#         return patient_image, patient_path

# if __name__ == '__main__':
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])])
    
#     dataset = TumorDataset(df='patient_for_hospital.csv', transform=transforms)
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, prefetch_factor=2)
#     import ipdb;ipdb.set_trace()
#     其他代码


import torch 
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import os 
import pandas as pd
from PIL import Image
import cv2

class TumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = pd.read_csv(df)
        self.patient = self.df['patient_id']
        self.label = self.df['label']
        self.censor = self.df['censor']
        self.time = self.df['months']
        self.transform = transform

    
    def __len__(self):
        return len(self.patient)


    def __getitem__(self, idx):
        patient = self.patient[idx]
        images_path = self.df[self.df['patient_id'] == patient]
        images = []
        for path in images_path['path']:
            try:
                # image = Image.open(path).convert('RGB')
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f'{path} has found error')
        patient_image = torch.stack(images)
        return patient_image, patient

if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])])
    
    dataset = TumorDataset(df='data_train_for_progosis_209.csv', transform=transforms)
    data_loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=8, prefetch_factor=2)
    import ipdb;ipdb.set_trace()



# import torch
# from torch.utils.data import Dataset
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
# import pandas as pd

# class TumorPipeline(Pipeline):
#     def __init__(self, df, batch_size, num_threads, device_id):
#         super(TumorPipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=42)
#         self.df = pd.read_csv(df)
#         self.input = ops.FileReader(file_root='', num_shards=1, shard_id=0, random_shuffle=True, pad_last_batch=True)
#         self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
#         self.resize = ops.Resize(device='gpu', resize_x=224, resize_y=224)
#         self.normalize = ops.CropMirrorNormalize(device='gpu',
#                                                  output_dtype=types.FLOAT,
#                                                  output_layout=types.NCHW,
#                                                  mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                                  std=[0.5 * 255, 0.5 * 255, 0.5 * 255])

#     def define_graph(self):
#         jpegs, labels = self.input(name='Reader')
#         images = self.decode(jpegs)
#         images = self.resize(images)
#         images = self.normalize(images)
#         return images, labels

# class TumorDALIDataset(Dataset):
#     def __init__(self, df, batch_size, num_threads, device_id):
#         self.pipe = TumorPipeline(df=df, batch_size=batch_size, num_threads=num_threads, device_id=device_id)
#         self.pipe.build()
#         self.epoch_size = self.pipe.epoch_size('Reader')

#     def __len__(self):
#         return self.epoch_size

#     def __iter__(self):
#         self.pipe.build()
#         self.dali_iter = DALIGenericIterator(self.pipe, ['images', 'labels'], size=self.epoch_size)
#         return iter(self.dali_iter)

# if __name__ == '__main__':
#     batch_size = 16
#     num_threads = 4
#     device_id = 0
#     dataset = TumorDALIDataset(df='patient_tumor_hospital.csv', batch_size=batch_size, num_threads=num_threads, device_id=device_id)
#     data_loader = DataLoader(dataset, batch_size=None, num_workers=num_threads)

#     for i, data in enumerate(data_loader):
#         images, labels = data['images'], data['labels']


        # 其他代码

# import os
# import cv2
# import lmdb
# import numpy as np
# from torch.utils.data import Dataset
# from tqdm import tqdm

# class LMDBDataset(Dataset):
#     def __init__(self, lmdb_path):
#         self.env = lmdb.open(lmdb_path, readonly=True)

#         with self.env.begin(write=False) as txn:
#             self.keys = [key.decode('utf-8') for key, _ in txn.cursor()]
#         import ipdb;ipdb.set_trace()
#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, index):
#         key = self.keys[index]

#         with self.env.begin(write=False) as txn:
#             img_bytes = txn.get(key.encode('utf-8'))

#         # 解析文件名，提取病人编号
#         patient_id = key.split('_')[0]

#         # 在这里你可以进行其他的预处理操作
#         image = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

#         return {'image': image, 'patient_id': patient_id}


# lmdb_path = 'lmdb'
# # 创建LMDB数据库
# # import ipdb;ipdb.set_trace()
# # import ipdb;ipdb.set_trace()
# # 创建数据集
# dataset = LMDBDataset(lmdb_path)
# import ipdb;ipdb.set_trace()
# # 打印数据集长度
# print(f"Dataset length: {len(dataset)}")

# # 通过索引访问数据
# sample = dataset[0]
# print(f"Patient ID: {sample['patient_id']}")
# print(f"Image shape: {sample['image'].shape}")
