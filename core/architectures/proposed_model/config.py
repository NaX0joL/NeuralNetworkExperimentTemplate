from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class ProposedModelConfig(ABSTRACT_Config):
    seq_len: int
    pred_len: int
    patch_len: int
    stride: int
        
    e_layers_num: int       # encoder layer num
    enc_in_feature: int     # encoder input feature
    d_layers_num: int       # decoder layer num
    dec_in_feature: int     # encoder input feature
    
    n_heads_num: int        # attention head number
    n_normal_heads: int     # normal attn number of head
    n_mp_attn_heads: int    # mp attn specific number of head
    qk_weight_share: bool   # flag to turn on q and k weight share
    d_model: int            # embedding vector dim
    d_ff: int               # feed forward dim
    
    dropout: float
    fc_dropout: float
    head_dropout: float
    attn_dropout: float
    
    use_pre_norm: bool
    
    attention_output_scaling: float = 1
    individual: int = 0
    padding_patch: str = None
    use_revin: bool = False
    use_affine: bool = False
    use_subtract_last: bool = False
    use_positional_encoding: bool = False
    decomposition: int = 0
    kernel_size: int = 25
    head_type: str = 'flatten'
    bottleneck_dim: int = 128
    
    @classmethod
    def default(self):
        proposed_model_config = self(
            seq_len = 500,
            pred_len = 2,
            patch_len = 25,
            stride = 1,
            
            e_layers_num = 1,
            enc_in_feature = 1,
            d_layers_num = 1,
            dec_in_feature = 1,
            
            n_heads_num = 1,
            n_normal_heads = 0,
            n_mp_attn_heads = 0,
            qk_weight_share = False,
            d_model = 256,
            d_ff = 512,
            
            dropout = 0.5,
            fc_dropout = 0.3,
            head_dropout = 0.1,
            attn_dropout = 0.1,
            
            use_pre_norm = False
        )
        return proposed_model_config