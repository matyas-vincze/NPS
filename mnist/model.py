import torch
from torch import nn
from typing import List
from torch.nn import functional as F

from dataclasses import dataclass, field
from utils import ImageEncoder, ImageDecoder, MLP, KeyQueryAttention, GroupLinear, argmax_onehot

@dataclass
class ModelArgs:
    # basic args
    cuda: bool = False
    seed: int = 21
    load_model: bool = False
    model_path: str = 'mnist.pth'
    wandb: bool = False
    # training args
    n_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.0001
    # model args
    n_operations: int = 4
    n_slots: int = 2
    n_rules: int = 4
    slot_size: int = 4
    dim_slot_embed: int = 128
    dim_rule_embed: int = 6
    dims_operation_encoder_hidden: List[int] = field(default_factory=lambda: [64,])
    dims_rule_mlp_hidden: List[int] = field(default_factory=lambda: [16,])
    dropout_prob: float = 0.1


class MnistSequentialNPS(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.image_encoder = ImageEncoder(args.dim_slot_embed)
        self.operation_encoder = MLP(input_dim=4, hidden_layer_dims=args.dims_operation_encoder_hidden, output_dim=args.dim_slot_embed)
        self.rule_network = RuleNetwork(args)
        self.image_decoder = ImageDecoder(args.dim_slot_embed)
        
        print(self)

    def forward(self, frames, operations):
        # frames shape            | (batch_size, n_operations, C=1, H=64, W=64)
        # operations shape        | (batch_size, n_operations, n_rules)
        
        batch_size, n_operations = frames.shape[:2]
        
        # --- STEP 1 --- #
        # encode image and operation into two slots M=2
        encoded_frames = self.image_encoder(frames.flatten(0, 1)).unsqueeze(1)
        encoded_operations = self.operation_encoder(operations.flatten(0, 1)).unsqueeze(1)
        # encoded_frame shape     | (batch_size*n_operations, 1, dim_slot_embed)
        # encoded_operation shape | (batch_size*n_operations, 1, dim_slot_embed)

        hidden = torch.cat((encoded_frames, encoded_operations), dim=1)
        hidden = hidden.view(batch_size, n_operations, *hidden.shape[1:])
        # hidden shape            | (batch_size, n_operations, 2, dim_slot_embed)

        # --- STEP 2-3-4 --- #
        rule_outputs = []
        for operation_idx in range(n_operations):
            rule_output = self.rule_network(hidden[:, operation_idx, ...])
            # rule_output shape   | (batch_size, 1, dim_rule_embed)
            rule_outputs.append(rule_output)
        rule_outputs = torch.stack(rule_outputs, dim=1)
        # rule_outputs shape      | (batch_size, n_operations, 1, dim_rule_embed)
        
        # decode final image
        output = self.image_decoder(rule_outputs.flatten(0, 1))
        # output shape            | (batch_size*n_operations, 1, 64, 64)

        return output
    

class RuleNetwork(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dropout = nn.Dropout(args.dropout_prob)

        self.rule_embeds = nn.Parameter(torch.randn(args.n_rules, args.dim_rule_embed))
        self.rule_mlps = nn.Sequential(
            GroupLinear(args.dim_slot_embed, 128, args.n_rules), 
            nn.Dropout(args.dropout_prob), nn.ReLU(), 
            GroupLinear(128, args.dim_slot_embed, args.n_rules))

        self.slot_rule_attention = KeyQueryAttention(
            dim_query = args.dim_rule_embed,
            dim_key = args.dim_slot_embed, 
            d_k = 32)
        self.primary_context_attention = KeyQueryAttention(
            dim_query = args.dim_rule_embed,
            dim_key = args.dim_slot_embed, 
            d_k = 16)
        
        self.rule_selection = []
        self.context_selection = []


    def forward(self, hidden):
        batch_size, n_slots, dim_slot_embed = hidden.size()
        n_rules, dim_rule_embed = self.rule_embeds.size()
        rule_embeds = self.rule_embeds[None, :, :].repeat(batch_size, 1, 1)

        # --- STEP 2 --- #
        # match the transformion embedding to the corresponding rule
        q1 = rule_embeds
        k1 = hidden
        slot_rule_attention = self.dropout(self.slot_rule_attention(q1, k1))
        slot_rule_attention_shape = slot_rule_attention.shape
        slot_rule_attention_flatten = slot_rule_attention.flatten(1)
        if self.training:
            slot_rule_attention_mask = F.gumbel_softmax(slot_rule_attention_flatten, tau=1.0, hard=True, dim=1)
        else:
            slot_rule_attention_mask = argmax_onehot(slot_rule_attention_flatten, dim=1)
        slot_rule_attention_mask = slot_rule_attention_mask.view(*slot_rule_attention_shape)  # (batch_size, num_rules, num_vars)

        rule_mask = slot_rule_attention_mask.sum(dim=2)
        selected_rule = (rule_embeds * rule_mask[:, :, None]).sum(dim=1)

        self.rule_selection.append(torch.argmax(rule_mask.detach(), dim=1).cpu().numpy())

        # --- STEP 3 --- #
        # select correct slot for rule application (image representation)
        q2 = selected_rule[:, None, :]
        k2 = hidden
        primary_context_attention = self.dropout(self.primary_context_attention(q2, k2))[:, 0, :]
        if self.training:
            context_mask = F.gumbel_softmax(primary_context_attention, tau=0.5, hard=True, dim=1)
        else:
            context_mask = argmax_onehot(primary_context_attention, dim=1)

        context_slot = (hidden * context_mask[:, :, None]).sum(dim=1)

        self.context_selection.append(torch.argmax(context_mask.detach(), dim=1).cpu().numpy())

        # --- STEP 4 --- #
        # apply the MLP of the selected rule to the selected context slot, then decode to get the transformed image

        input = context_slot[:, None, :].repeat(1, n_rules, 1)
        output = self.rule_mlps(input)
        output = (output * rule_mask[:, :, None]).sum(dim=1)
        return output

    def reset(self):
        self.rule_selection.clear()
        self.context_selection.clear()
