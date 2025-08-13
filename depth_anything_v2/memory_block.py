import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .memory_bank import MemoryBank
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention


class MemoryBlock(nn.Module):
    def __init__(self, memory_channel, max_memory_length, num_attention_layers):
        super().__init__()
        self.max_memory_length = max_memory_length
        
        num_heads = memory_channel // 64
        cross_attention = RoPEAttention(
            embedding_dim=memory_channel,
            num_heads=num_heads,
            dropout=0.1,
            rope_k_repeat=True
        )
        
        self_attention = RoPEAttention(
            embedding_dim=memory_channel,
            num_heads=num_heads,
            dropout=0.1,
            rope_k_repeat=False, 
        )
        
        memory_attention_layer = MemoryAttentionLayer(
            activation='gelu',
            cross_attention=cross_attention,
            d_model=memory_channel,
            dim_feedforward=memory_channel * 2,
            dropout=0.1,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=True,
            self_attention=self_attention
        )
        
        self.memory_attention= MemoryAttention(
            d_model=memory_channel,
            pos_enc_at_input=True,
            layer=memory_attention_layer,
            num_layers= num_attention_layers,
            batch_first = True
        )
        self.memory_bank = MemoryBank(maxlen=max_memory_length)

        self.curr_pos_enc = nn.Parameter(torch.zeros(1, 1, memory_channel))
        trunc_normal_(self.curr_pos_enc, std=0.02)
        # todo
        # self.curr_pos_enc = PositionEmbeddingSine(num_pos_feats=memory_channel)
        # self.curr_tpos_enc = nn.Parameter(torch.zeros(1, 1, memory_channel))
        # self.curr_spos_enc = nn.Parameter(torch.zeros(1, 1, memory_channel))
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(1, max_memory_length, memory_channel))
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, memory_channel))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        trunc_normal_(self.no_mem_embed, std=0.02)
        
        # self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, memory_channel))
        # trunc_normal_(self.no_mem_pos_enc, std=0.02)
    
        # TODO: 메모리 인코더 수정  in channel, out channel 직접설정하도록
        self.memory_encoder = MemoryEncoder(
            out_dim=memory_channel,
            mask_downsampler= nn.Sequential(
                MaskDownSampler(embed_dim=1, kernel_size=3, stride=2, padding=1, total_stride=2),
                MaskDownSampler(embed_dim=1, kernel_size=7, stride=7, padding=0, total_stride=7),
            ),  #(embed_dim=128, kernel_size=3, stride=2, padding=1),
            fuser=Fuser(layer=CXBlock(memory_channel), num_layers=2),
            position_encoding=PositionEmbeddingSine(num_pos_feats=memory_channel),
            in_dim=memory_channel,
        )
        
    def clear_memory(self):
        self.memory_bank.clear_memory()

    # channel 1024, 128??        
    def update_memory(self, img_feature, depth):
        B, HW, C = img_feature.shape
        H, W = int(HW ** 0.5), int(HW ** 0.5)  # todo input x로부터 구하도록 수정
        img_feature_permute = img_feature.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        encoded_memory_dict = self.memory_encoder(img_feature_permute, depth)
        encoded_memory_dict["memory_feature"] = encoded_memory_dict["memory_feature"].reshape(B, C, HW).permute(0, 2, 1).contiguous()
        encoded_memory_dict["memory_pos_enc"] = encoded_memory_dict["memory_pos_enc"].reshape(B, C, HW).permute(0, 2, 1).contiguous()
        self.memory_bank.update_memory(encoded_memory_dict)
    
    def forward(self, img_feature):
        memories = self.memory_bank.get_memory()
        if memories:
            to_cat_memory = []
            to_cat_memory_pos_embed = []
            S = len(memories)
            for i, memory in enumerate(memories):
                memory_feature = memory["memory_feature"].to(img_feature.device)
                memory_pos_enc = memory["memory_pos_enc"].to(img_feature.device)
                memory_pos_enc = memory_pos_enc + self.maskmem_tpos_enc[:, self.max_memory_length - S + i]
                
                to_cat_memory.append(memory_feature)
                to_cat_memory_pos_embed.append(memory_pos_enc)
            
            memory = torch.cat(to_cat_memory, dim=1)
            memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=1)
            img_feature = self.memory_attention.forward(
                img_feature, 
                memory, 
                self.curr_pos_enc,
                memory_pos_embed,
                0
            )
        else:
            B, HW = img_feature.shape[0], img_feature.shape[1]
            img_feature = self.memory_attention.forward(
                img_feature,
                self.no_mem_embed.expand(B, HW, -1), 
                self.curr_pos_enc,
                self.maskmem_tpos_enc[:, self.max_memory_length - 1],
                0
            )
        
        return img_feature
