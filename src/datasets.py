import os
import pandas as pd
import torch
from transformers import AutoProcessor, AltCLIPModel
from PIL import Image,ImageFile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import str2bool, generate_name

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MemesDataset(Dataset):
    def __init__(self, root_folder, dataset, split='train', image_size=224):
        super(MemesDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split

        self.image_size = image_size

        self.info_file = os.path.join(root_folder, dataset, f'labels/{dataset}_finally.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]


        if self.dataset == 'irony_handle':
            image_fn = row['image'].split('/')[1]
        else:
            image_fn = row['image']
        image = Image.open(f"{self.root_folder}/{self.dataset}/img/{image_fn}").convert('RGB') \
            .resize((self.image_size, self.image_size))
     

        item = {
            'image': image,
            'text': row['text'],
            'label': row['label'],
            'inference_text': row['inference_text']
        }

        return item


class MemesCollator(object):
    def __init__(self, args):
        self.args = args
    def __call__(self, batch):


        labels = torch.LongTensor([int(item['label']) for item in batch])


        texts= []         
        imgs = []         
        masks=[]          
        
        clip_preprocess =AutoProcessor.from_pretrained(" ")

        img = []
        texts = []
        gos=[]
       
        for item in batch:
            if self.args.infer:
                texts.append(item['inference_text'])
            else:
                texts.append(item['gosm'])
            pixel_values = clip_preprocess(images=item['image'], return_tensors='pt')
            imgs.append(pixel_values['pil'])
    
        texts=clip_preprocess(text=texts,padding=True,truncation=True, max_length=77,return_tensors='pt')

       
        batch_new=texts
        batch_new['labels'] = labels
      
        pixel_values = torch.cat([item for item in imgs], dim=0)
        batch_new['pil'] = pixel_values
        return batch_new


def load_dataset(args, split):
    dataset = MemesDataset(root_folder=f'../resources/datasets', dataset=args.dataset, split=split,
                           image_size=args.image_size)
    return dataset


