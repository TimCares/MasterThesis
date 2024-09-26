import torch

def ensure_attention_to_self(attn_mask, padding_mask):
    B, N, _ = attn_mask.shape
    self_attn_indicator = torch.eye(N, device=attn_mask.device).expand(B, -1, -1)
    self_attn_indicator = self_attn_indicator.bool() & ~padding_mask.unsqueeze(1).expand(-1, N, -1).bool()
    attn_mask[self_attn_indicator] = 0 # all tokens can attend to themselves except for padding tokens
    return attn_mask

def generate_attention_mask(mask, padding_mask):
    """Masked tokens only attend to themselves"""
    _, N = mask.shape

    attn_mask = mask.long() | padding_mask.long()
        
    # mask query-wise | mask key-wise
    attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, N) | attn_mask.unsqueeze(1).expand(-1, N, -1)

    attn_mask = ensure_attention_to_self(attn_mask, padding_mask)

    return attn_mask

def generate_attention_mask_unmasked_token_access(mask, padding_mask):
    """Masked tokens have access to unmasked tokens, but not vice versa"""
    _, N = mask.shape

    attn_mask = padding_mask.unsqueeze(2).expand(-1, -1, N).long() | padding_mask.unsqueeze(1).expand(-1, N, -1).long()

    m_ = mask.unsqueeze(1).expand(-1, N, -1).long()

    attn_mask = attn_mask | m_

    attn_mask = ensure_attention_to_self(attn_mask, padding_mask)

    return attn_mask