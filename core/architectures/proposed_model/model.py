import torch
from torch import nn, Tensor
from typing import Optional

from .config import ProposedModelConfig
from .PatchTST.backbone import PatchTST_backbone
from .series_decomposition import series_decomp



class ProposedModel(nn.Module):
    def __init__(self, configs:ProposedModelConfig, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=True, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, 
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False, **kwargs):
        
        super().__init__()
        
        self.configs = configs
        
        if hasattr(configs, "n_normal_heads"):
            n_normal_heads = configs.n_normal_heads
        else:
            n_normal_heads = 0
        
        if hasattr(configs, "n_mp_attn_heads"):
            n_mp_attn_heads = configs.n_mp_attn_heads
        else:
            n_mp_attn_heads = 0
            
        if hasattr(configs, "qk_weight_share"):
            qk_weight_share = configs.qk_weight_share
        else:
            qk_weight_share = False
        
        c_in = configs.enc_in_feature
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers_num
        n_heads = configs.n_heads_num
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        attn_dropout = configs.attn_dropout
        
        pre_norm = configs.use_pre_norm
        attention_output_scaling = configs.attention_output_scaling
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.use_revin
        affine = configs.use_affine
        subtract_last = configs.use_subtract_last
        
        use_positional_encoding = configs.use_positional_encoding
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        head_type = configs.head_type
        bottleneck_dim = configs.bottleneck_dim
        
        self.res_attention = res_attention
        
        # model definition
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
        
        self.softmaxed_attn_score = []
        self.attn_score = []
        
        self.quick_classification_head = QuickClassificationHead(configs.seq_len, configs.enc_in_feature, configs.pred_len)
        
        return
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        self.softmaxed_attn_score.clear()
        self.attn_score.clear()
        
        # print("model input shape:", x.shape) 
        # print("model input std:", x.std().item()) 
        
        x = QuickInputOutputFixxer.fix_input(x)
        
        #print("model input after shape:", x.shape) 
        
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            
            # # manually save attention matrix
            # self.softmaxed_attn_score = self.model.softmaxed_attn_score
            # if self.res_attention:
            #     self.attn_score = self.model.attn_score
        
        x = QuickInputOutputFixxer.fix_output(x)
        
        # print("model output std:", x.std().item())
        # print("model output shape:", x.shape) 
        
        return x



class QuickInputOutputFixxer():
    
    @classmethod
    def _checkInputShape(self, input:Tensor) -> None:
        print(input.shape)
        return
    
    @classmethod
    def fix_input(self, input:Tensor) -> Tensor:
        input = input.unsqueeze(-1)
        return input
    
    @classmethod
    def fix_output(self, output:Tensor) -> Tensor:
        output = output.squeeze(-1)
        return output



class QuickClassificationHead(nn.Module):

    def __init__(self, input_length, channel, n_class, dropout:float=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(input_length * channel, n_class)
        return
    
    def forward(self, x:Tensor) -> Tensor:   # x: [Batch, Input length, Channel]
        x = torch.flatten(x, start_dim=1)           # x: [Batch, Input length * Channel]
        x = self.dropout(x)
        print(x.shape)
        x = self.linear(x)                          # x: [Batch, n_class]
        
        return  x