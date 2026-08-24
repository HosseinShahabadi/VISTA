"""T5 text-encoder self-attention capture.

Used to inspect how the prompt encoder itself binds attributes to nouns, before
any image-side attention exists. Enabled via
`InfinityWrapper.generate(t5_attention=True)`.
"""

import torch
import torch.nn.functional as F


class T5AttentionStore:
    def __init__(self):
        self.attn = {}

    def add(self, layer_idx, attn):
        self.attn[layer_idx]= attn.detach().cpu()

    def get(self):
        return self.attn
    
def t5_attn_hook(layer_idx, store):
    def hook(module, inputs, outputs):
        if not isinstance(outputs, tuple):
            return

        # T5LayerSelfAttention outputs:
        #
        # 0 -> hidden states
        # 1 -> present_key_value_state (sometimes)
        # 2 -> position bias
        # 3 -> attention weights
        #
        # OR depending on HF version:
        # 0 -> hidden states
        # 1 -> attention weights
        # 2 -> position bias

        attn= None
        for out in outputs:
            if torch.is_tensor(out):
                # attention weights should be:
                # [B, H, L, L]
                if out.dim() == 4:
                    B, H, L1, L2 = out.shape
                    if L1==L2:
                        # probabilities should be <=1
                        if out.max() <= 1.0+1e-3:
                            attn= out
                            break
        if attn is not None:
            store.add(layer_idx, attn.detach().cpu())
    return hook
    
# def t5_attn_hook(layer_idx, store):
#     def hook(module, inputs, outputs):
#         if isinstance(outputs, tuple):
#             if len(outputs)>1:
#                 attn= outputs[1]
#                 if attn is not None:
#                     store.add(layer_idx, attn)
#     return hook


def register_t5_attention_hooks(t5_encoder, store):
    hooks= []
    # for i, layer in enumerate(t5_encoder.encoder.block):
    #     h= layer.layer[0].SelfAttention.register_forward_hook(t5_attn_hook(i, store))
    #     hooks.append(h)
    for i, layer in enumerate(t5_encoder.encoder.block):
        h = layer.layer[0].register_forward_hook(
            t5_attn_hook(i, store)
        )
        hooks.append(h)
    
    return hooks


def wrong_binding_score(attn_layer, tok_color1, tok_obj1, tok_color2):
    a= attn_layer.mean(0)
    correct= a[tok_obj1, tok_color1]
    incorrect= a[tok_obj1, tok_color2]
    print("correct:", correct.item())
    print("incorrect:", incorrect.item())
    return (incorrect/(correct+incorrect+1e-8)).item()



def normalize_token(tok):
    """
    Normalize SentencePiece tokens.
    
    Examples:
    ▁red   -> red
    ▁blue  -> blue
    bird   -> bird
    </s>   -> </s>
    """
    tok= tok.replace("▁", "")
    return tok.strip().lower()


def find_token_positions(decoded_tokens, target_word):
    """
    Finds token indices corresponding to a word,
    even if the word is split into multiple subwords.

    Returns:
        list[int]
    """
    target_word= target_word.lower().strip()
    normalized= [normalize_token(t) for t in decoded_tokens]

    # -------------------------------------------------
    # 1. Exact single-token match
    # -------------------------------------------------
    for i, tok in enumerate(normalized):
        if tok == target_word:
            return [i]
    # -------------------------------------------------
    # 2. Multi-token merge search
    # -------------------------------------------------
    for start in range(len(normalized)):
        merged= ""
        for end in range(start, len(normalized)):
            merged+= normalized[end]
            if merged==target_word:
                return list(range(start, end+1))
            if len(merged)>len(target_word):
                break
    return []


def hidden_state_binding_score(hidden, tok_color1, tok_obj1, tok_color2):
    """
    hidden:
        [B, L, D]

    Measures whether OBJ1 embedding is
    closer to WRONG color than CORRECT color.
    """
    obj1= hidden[:, tok_obj1, :]
    c1= hidden[:, tok_color1, :]
    c2= hidden[:, tok_color2, :]

    sim_correct= F.cosine_similarity(obj1, c1, dim= -1)
    sim_wrong= F.cosine_similarity(obj1, c2, dim= -1)

    score= sim_wrong/(sim_correct+sim_wrong+1e-8)

    return score.mean().item()