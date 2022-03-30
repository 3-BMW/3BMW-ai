import random
import os
import argparse
import wandb

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from tqdm import tqdm

from model import ImageModel
from dataset import TrainDataset, num2label, label2num

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default = './best_model/fold') 
    parser.add_argument('--wandb_path', type= str, default= 'effb0-base')
    parser.add_argument('--train_path', type= str, default= './data/resize_train.csv')
    parser.add_argument('--model', type=str, default='efficientnet_b0')
    parser.add_argument('--loss', type=str, default= 'CE')

    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gradient_accum', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=231)
    parser.add_argument('--step_size', type=int, default=200)

    
    args= parser.parse_args()

    return args

def train(train_loader, valid_loader, model, optimizer, criterion, args):

    train_loss, train_acc= 0, 0
    valid_loss, valid_acc= 0, 0

    for epoch in range(args.epochs):

        pbar= tqdm(enumerate(train_loader), total= len(train_loader))
        for step, (images, labels) in pbar:
            model.train()

            images.to(device)
            labels.to(device)

            optimizer.zero_grad()

            outputs= model(images)
            loss= criterion(outputs, labels)

            _, preds= torch.max(outputs, 1)
            # print(preds)
            # print(labels)
            print(torch.sum(preds== labels.data))
            loss.backward()
            optimizer.step()

            train_loss+= loss.item()
            train_acc+= torch.sum(preds== labels.data)

            if (step+1) % args.step_size == 0:

                model.eval()
                with torch.no_grad():
                    pbar= tqdm(enumerate(valid_loader), total= len(valid_loader))
                    for idx, (images, labels) in pbar:
                        images.to(device)
                        labels.to(device)

                        outputs= model(images)

                        _, preds= torch.max(outputs, 1)
                        loss= criterion(outputs, labels)

                        valid_acc += torch.sum(preds == labels.data)
                        valid_loss += loss.item()

                    train_loss= train_loss/ args.step_size
                    train_acc= train_acc/ (args.step_size * args.batch_size)
                    valid_loss= valid_loss/ args.step_size
                    valid_acc= valid_acc/ (args.step_size * args.batch_size)

                    wandb.log({'train/accuracy': train_acc, 'valid/accuracy': valid_acc,
                    'train/loss': train_loss, 'valid/loss': valid_loss, }) 

                    print(f'epoch: {epoch}, train loss: {train_loss}, train acc: {train_acc} \
                        valid loss: {valid_loss}, valid acc: {valid_acc}')

                    train_loss, train_acc= 0, 0
                    valid_loss, valid_acc= 0, 0

        # train_loss= train_loss/ len(train_loader)
        # train_acc= train_acc/ len(train_loader.dataset)
        # valid_loss= valid_loss/ len(valid_loader)
        # valid_acc= valid_acc/ len(valid_loader.dataset)  

        # wandb.log({'train/accuracy': train_acc, 'valid/accuracy': valid_acc,
        # 'train/loss': train_loss, 'valid/acc': valid_loss, }) 

        # print(f'epoch: {epoch}, train loss: {train_loss}, train acc: {train_acc} \
        #     valid loss: {valid_loss}, valid acc: {valid_acc}')

if __name__ == '__main__':
    args= get_config()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    seed_everything(args.seed)

    data= pd.read_csv(args.train_path)
    data= label2num(data)

    print(f'data length : {len(data)}')
    train_df, valid_df= train_test_split(data, test_size= 0.2, shuffle= True, random_state= args.seed)
    print(f'train data length : {len(train_df)}')
    print(f'valid data length : {len(valid_df)}')

    train_transform= transforms.Compose([
        transforms.Resize((32, 32), interpolation= 2),
        transforms.ToTensor()
    ])

    trainset= TrainDataset(train_df, transforms= train_transform)
    validset= TrainDataset(valid_df, transforms= train_transform)

    train_loader= DataLoader(trainset, batch_size= args.batch_size, shuffle= True)
    valid_loader= DataLoader(validset, batch_size= args.batch_size, shuffle= True)

    model= ImageModel(args.model, args.num_classes)
    optimizer= torch.optim.Adam(model.parameters(), lr= args.lr)
    criterion= torch.nn.CrossEntropyLoss()

    print(f'train start')
    run= wandb.init(project= 'bmw', entity='rockmiin',name= args.wandb_path)
    train(train_loader, valid_loader, model, optimizer, criterion, args)
    run.finish()
    print(f'train finish')






