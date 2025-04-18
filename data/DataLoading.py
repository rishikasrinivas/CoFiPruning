import torch
import os


def load_masks(cluster_num, directory):
    act_mask_sents=[]
    print(directory)
    for file in os.listdir(directory):
        file= directory + "/" + file
        act_mask_sent=torch.load(file)
        act_mask_sents.append(act_mask_sent)
    act_masks=torch.stack(act_mask_sents,dim=0)
    return act_masks