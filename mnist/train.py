import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from einops import rearrange

from utils import count_mapping


def train_epoch(model, dataloader, optim, metric, device):
    model.train()

    pbar = tqdm(dataloader)
    loss_list = []
    rule_gold, rule_selected, slot_selected = [], [], []

    for batch_idx, batch in enumerate(pbar):
        images, operations = batch[0].to(device), batch[1].to(device)

        model.rule_network.reset()
        optim.zero_grad()

        outputs = model(images[:, :-1, ...], operations)
        test = images[:, 1:, ...].flatten(0, 1)
        
        loss = metric(outputs, images[:, 1:, ...].flatten(0, 1))
        loss_list.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # TODO
        optim.step()
        
        rule_gold.append(operations.detach().flatten(0, 1).argmax(dim=1).cpu())
        rule_selected.append(np.array(model.rule_network.rule_selection))
        slot_selected.append(np.array(model.rule_network.context_selection))
        
    rule_gold = np.concatenate(rule_gold, axis=0)
    rule_selected = rearrange(rule_selected, 'n t b -> (n b t)')
    slot_selected = rearrange(slot_selected, 'n t b -> (n b t)')
    
    selected_rules_count = count_mapping(rule_gold, rule_selected, range(4))
    selected_slots_count = count_mapping(rule_selected, slot_selected, range(4))

    return np.mean(loss_list), selected_rules_count, selected_slots_count
