import models
import torch
import util
import tqdm as tqdm
from contextlib import nullcontext
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np
#!pip install llm2vec #add to donwloads
import os
from models import NLI_models as models
from tqdm import tqdm
from data.snli import SNLI, pad_collate
from collections import defaultdict
import os
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
def create_dataloaders(max_data):
    root_dir=f"DataLoaders/"
    os.makedirs(root_dir, exist_ok=True)
    if not ('train_dataset.pth' in os.listdir(root_dir) or 'val_dataset.pth' in os.listdir(root_dir) and 'test_dataset.pth' in os.listdir(root_dir)):
        train = SNLI("data/snli_1.0", "train", max_data=None)
        train_loader = DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(train_loader.dataset, f'{root_dir}/train_dataset.pth')
        
        val = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train.stoi, train.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader.dataset, f'{root_dir}/val_dataset.pth')
        
        test = SNLI("data/snli_1.0", "test", max_data=max_data, vocab=(train.stoi, train.itos), unknowns=True)
        test_loader = DataLoader(
            test,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(test_loader.dataset, f'{root_dir}/test_dataset.pth')
    else:
        train_dataset = torch.load(f'{root_dir}/train_dataset.pth')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=100, 
            shuffle=True, 
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        val = SNLI("data/snli_1.0","dev",max_data=10000,vocab=(train_loader.dataset.stoi, train_loader.dataset.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader.dataset, f'{root_dir}/val_dataset.pth')
        
        
        test_dataset = torch.load(f'{root_dir}/test_dataset.pth')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        
    
    dataloaders = {
        'train': train_loader,
        'val':val_loader,
        'test': test_loader
    }
    return train_loader.dataset, val_loader.dataset,test_loader.dataset, dataloaders


def run(split, epoch, model,model_type, optimizer, criterion, dataloader, total_epochs, device='cuda'):
    training = split == "train"
    if training:
        ctx = nullcontext
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()

    ranger = tqdm(dataloader[split], desc=f"{split} epoch {epoch}")
    
    if model_type in ['bert', 'llama']:
        total_steps = len(dataloader['train']) *total_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    

    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()

    #for (s1, s1len, s2, s2len, targets) in ranger:
    for inputs in ranger:
        print(inputs)
        model(inputs)
        return
        if device == 'cuda':
            s1 = s1.cuda()
            s1len = s1len.cuda()
            s2 = s2.cuda()
            s2len = s2len.cuda()
            targets = targets.cuda()

    

        batch_size = targets.shape[0]
        
        with ctx():
            logits = model(s1, s1len, s2, s2len)
            loss = criterion(logits, targets)
 
        if training:
            optimizer.zero_grad()
            loss.backward()
            for layer in model.layers:
                layer.pruning_mask.cuda()
                if layer.weights.grad is not None:
                    layer.weights.grad *= layer.pruning_mask.to(device)
            
        
            optimizer.step()
            
                
        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        
        if model_type in ['bert', 'llama']:
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            scheduler.step()

        ranger.set_description(
            f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}"
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

def finetune_pruned_model(model,model_type, optimizer,criterion, train, val, dataloaders, finetune_epochs, prune_metrics_dir,device):
    metrics = {"best_val_acc": 0.0, "best_val_epoch": 0, "best_val_loss": np.inf, "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    
    for epoch in range(finetune_epochs):
        train_metrics = run(
            "train", epoch, model, model_type, optimizer, criterion, dataloaders, finetune_epochs, device
        )

        val_metrics = run(
            "val", epoch, model, model_type, optimizer, criterion, dataloaders, finetune_epochs, device
        )

        for name, val in train_metrics.items():
            metrics[f"train_{name}"].append(val)

        for name, val in val_metrics.items():
            metrics[f"val_{name}"].append(val)

        is_best = val_metrics["acc"] > metrics["best_val_acc"]

        if is_best:
            metrics["best_val_epoch"] = epoch
            metrics["best_val_acc"] = val_metrics["acc"]
            metrics["best_val_loss"] = val_metrics["loss"]
            #fileio.log_to_csv(os.path.join(prune_metrics_dir,"pruned_status.csv"), [epoch, val_metrics["acc"], val_metrics["loss"]], ["EPOCH", "ACCURACY", "LOSS"])
        
       
        util.save_metrics(metrics, prune_metrics_dir)
        util.save_checkpoint(serialize(model, model_type, train), is_best, prune_metrics_dir)
        if epoch % 1 == 0:

            util.save_checkpoint(
                serialize(model, model_type, train), False, prune_metrics_dir, filename=f"LotTick{epoch}.pth"
            )
        
    path_to_ckpt = os.path.join(prune_metrics_dir, f"model_best.pth")
    print(f"Loading best weights from {path_to_ckpt}")
    model.load_state_dict(torch.load(path_to_ckpt)['state_dict'])
    
    return model


def build_model(vocab_size, model_type, vocab, embedding_dim=300, hidden_dim=512, device='cuda'):
    """
    Build a bowman-style SNLI model
    """
    
    if model_type=='bert':
        model=models.BertEntailmentClassifier(vocab=vocab, device=device)
    elif model_type == 'bowman':
        enc = models.TextEncoder(
            vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim
        )
        model = models.BowmanEntailmentClassifier(enc, device)
    elif model_type == 'llama':
            if vocab is None:
                raise Exception('Llama model requires passing the datasets vocab field')
            model = models.LLAMAEntailmentClassifier(vocab=vocab,freeze_encoder=True)
    return model

def load_model(max_data, model_type, train, store_dir = None, ckpt=None, device='cuda'):
    model = build_model(vocab_size=len(train.stoi), model_type=model_type, vocab={'stoi': train.stoi, 'itos': train.itos}, embedding_dim=300, hidden_dim=512, device=device)
    
    
    if ckpt:
        if type(ckpt) == str:
            print(f"Loading from {ckpt} with model {model_type}")
            ckpt_ = torch.load(ckpt, map_location = torch.device(device))
        model.load_state_dict(ckpt_["state_dict"])
    else:
        print("train itos = ", len(train.stoi))
        util.save_checkpoint( 
                serialize(model, model_type, train), False, store_dir, filename=  f"{model_type}_random_inits.pth"
        )
        ckpt = os.path.join(store_dir, f"{model_type}_random_inits.pth")
            
        
    return model, ckpt


def serialize(model,model_type, dataset):
    if model_type == ['llama','bert']:
        return {
            "encoder_name": model.encoder_name, 
            "state_dict": model.state_dict(), 
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        }
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }

def run_eval(model, val_loader):
    model.cuda()
    model.eval()
    all_preds = []
    all_targets = []
    for (s1, s1len, s2, s2len, targets) in val_loader:
        if settings.CUDA:
            s1 = s1.cuda()
            s1len = s1len.cuda()
            s2 = s2.cuda()
            s2len = s2len.cuda()

        with torch.no_grad():
            logits = model(s1, s1len, s2, s2len)

        preds = logits.argmax(1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_targets = np.concatenate(all_targets, 0)
    acc = (all_preds == all_targets).mean()
    return np.round(acc,3)