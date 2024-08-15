import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import clip
from transformers import AltCLIPModel
import torch.nn.functional as F



CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ = 768


class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super(LinearProjection, self).__init__()
      
        map_layers = [nn.Linear(input_dim, output_dim),
                      nn.Dropout(p=drop_probs[0])]

        for _ in range(1, num_layers):
            map_layers.extend(
                [nn.ReLU(), nn.Linear(output_dim, output_dim), nn.Dropout(p=drop_probs[0])])

        self.proj = nn.Sequential(*map_layers)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        return self.proj(x)


class mmeetsClassifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.dataset = args.dataset
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.fusion = args.fusion
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size

        self.name = args.name



        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')


        self.model = AltCLIPModel.from_pretrained("").to('cuda')

        if self.name in ['hate-clipper', 'adaptation']:
            if args.fusion == 'align':
                pre_output_input_dim = self.map_dim
            elif args.fusion == 'concat':
                pre_output_input_dim = self.map_dim * 2
        else :
            pre_output_input_dim = self.map_dim

        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        if self.num_pre_output_layers >= 1:
            for _ in range(1, self.num_pre_output_layers):
                pre_output_layers.extend(
                    [nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim


        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(output_input_dim, 1)


        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, batch):
        pass

    def common_step(self, batch):





        for param in self.model.text_model.parameters():
            param.requires_grad = False
   
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

     
        pixel_values, texts,masks = batch['pixel_values'],batch['input_ids'],batch['attention_mask']
        text_masks={'input_ids':texts,'attention_mask':masks}
        pixel_values ={'pixel_values':pixel_values}

    
        image_features =self.model.get_image_features(**pixel_values)
  
        text_features = self.model.get_text_features(**text_masks)

        output = {}

   

        if self.name in ['hate-clipper', 'adaptation']:
         
            image_features = F.normalize(image_features, p=2, dim=1)  

            text_features = F.normalize(text_features, p=2, dim=1)  

            if self.fusion == 'align':
                features = torch.mul(image_features, text_features)
            elif self.fusion == 'concat':
                features = torch.cat([image_features, text_features], dim=1)
            else:
                raise ValueError()
        elif self.name=='text-only':
            features = F.normalize(text_features, p=2, dim=1)
        elif self.name=='image-only':
            features = F.normalize(image_features, p=2, dim=1)
        else:
            raise ValueError()
        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  
        preds_proxy = torch.sigmoid(logits)

        preds = (preds_proxy >= 0.5).long()
        output['preds']=preds
        output['labels']=batch['labels']
        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])

        output['f1']=self.f1(preds, batch['labels'])
        print(preds)
        print(batch['labels'])
        print(output['f1'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']
        self.log('train/loss', output['loss'], on_epoch=False,on_step=True)
        self.log('train/acc', output['accuracy'], on_epoch=True,on_step=True)
        self.log('train/f1', output['f1'], on_epoch=True,on_step=True)
        print('xxxxs')

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.acc.update(output['preds'], output['labels'])
        self.f1.update(output['preds'], output['labels'])
        self.log('val/loss_step', output['loss'],on_step=True,on_epoch=False)
        self.log('val/acc', output['accuracy'],on_step=True,on_epoch=True)
        self.log('val/f1', output['f1'],on_step=True,on_epoch=True)
        print('val')


    def test_step(self, batch, batch_idx):
        if self.dataset == 'zh':
            prefix_map = {
                0: 'val',
                1: 'test',
            }
        elif self.dataset == 'en':
            prefix_map = {
                0: 'val',
                1: 'test',
            }
        else:
            raise ValueError()



        output = self.common_step(batch)
        self.acc.update(output['preds'], output['labels'])
        self.f1.update(output['preds'], output['labels'])


    def training_epoch_end(self, outputs):
        self.acc.reset()
        self.f1.reset()
        print('train_fally')

    def validation_epoch_end(self, outputs):
        self.acc.reset()
        self.f1.reset()
        print('val_fally')

    def test_epoch_end(self, outputs):
        self.log('test/acc', self.acc.compute())
        self.log('test/f1', self.f1.compute())
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()
    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

def create_model(args):
    model =mmeetsClassifier(args=args)
    return model