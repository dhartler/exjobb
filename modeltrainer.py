import torch
import torch.nn
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import wandb
from rotary_embedding_torch import RotaryEmbedding
import torch.nn as nn
import numpy as np
import math
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)

from gumbel_softmax import interpolate_vectors


model_name = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
# model_name = "./1.3b_inst_alibi_4k_wte_locked"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# tokenizer = AutoTokenizer.from_pretrained(model_name)
wandb_log = True
alibi_enabled = True
rope_enabled = False
CTX_LEN = 4096
INTERPOLATION_FACTOR = 1
TRAIN_DATA = "./data/inst_10m_tok_4096_train.pt"
EVAL_DATA = "./data/inst_10m_tok_4096_eval.pt"
OUT_DIR = "./1.3b_inst_alibi_4k_wte_locked"

doing_inference=False

# torch.backends.cuda.sdp_kernel.enable_flash = True           # Only supported on A100 or H100 
# torch.backends.cuda.sdp_kernel.enable_mem_efficient = False    # Supported for Quadro RTX 5000 
# torch.backends.cuda.sdp_kernel.enable_math = False            # Supported for Quadro RTX 5000 

class CustomAttn(nn.Module):
    def __init__(self, config, 
                 head_id, 
                 c_attn, 
                 c_proj, 
                 attn_dropout, 
                 resid_dropout, 
                 is_cross_attention=False):
        super().__init__()

        #max_positions = config.max_position_embeddings
        if rope_enabled: 
            max_positions = CTX_LEN*INTERPOLATION_FACTOR
        else:
            max_positions = CTX_LEN
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = True# hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )

        ##### alibi offset and head_id is for alibi head scaling #####
        if alibi_enabled:
            self.register_buffer(
            'alibi_offset',
            torch.tensor(np.array([[i for i in range(-row, 1)] + [0]*(max_positions-1-row)  for row in range(max_positions)]), dtype=torch.bfloat16)
            )
            self.m = 1/((2**( (8/config.n_head) * (head_id+1) )))
        ###############################

        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        ## Just copy trained layers from previous attention block
        self.c_attn = c_attn
        self.c_proj = c_proj
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout

        # Implementation of RoPE
        if rope_enabled:
            self.rotary_embedding = RotaryEmbedding(dim=self.head_dim)
        else:
            self.rotary_embedding = None
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        

        if self.flash and not doing_inference:
            # efficient attention using Flash Attention CUDA kernels
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=model.config.attn_pdrop if self.training else 0, is_causal=True)
            return y, None
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            if self.scale_attn_weights:
                attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

            if not self.is_cross_attention:
                # if only "normal" attention layer implements causal mask
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
                attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights = attn_weights + attention_mask

            # Apply ALiBi mask
            if alibi_enabled:
                attn_weights = attn_weights[:, :] + self.alibi_offset[key_length - query_length : key_length, :key_length]*self.m
            
            attn_weights = nn.Softmax(dim=-1)(attn_weights)
            attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            
            attn_output = torch.matmul(attn_weights, value)

            return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        if rope_enabled:
            query = self.rotary_embedding.rotate_queries_or_keys(query)
            key = self.rotary_embedding.rotate_queries_or_keys(key)
        
        
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)




        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
        # Shift input_ids by one position to create the labels
        # Ensuring labels are shifted and the last position is ignored or masked
        self.labels = self.input_ids[:, 1:].contiguous()

    def __getitem__(self, idx):
        # For input_ids, take all tokens except the last one
        input_data = self.input_ids[idx, :-1]
        # For labels, all tokens are shifted by one, dropping the first token
        label_data = self.labels[idx]
        return {"input_ids": input_data, "labels": label_data}

    def __len__(self):
        return self.input_ids.size(0)


def interpolate_pos_emb():
    # print(model.transformer.wpe.weight.data.shape[1])
    # interp_len = model.transformer.wpe.weight.data.shape[0]*2
    # pos_emb_2x = torch.nn.Embedding(interp_len, model.transformer.wpe.weight.data.shape[1], dtype=torch.bfloat16).to(device)
    # pos_emb_2x.weight.data[range(0,interp_len, 2)] = model.transformer.wpe.weight.data
    # for i in range(1, interp_len-1, 2):
    #     pos_emb_2x.weight.data[i] = interpolate_vectors(pos_emb_2x.weight.data[[i-1, i+1]], 0.5, device=device)
    #     # pos_emb_2x.weight.data[i] = 0.5*pos_emb_2x.weight.data[i-1] + 0.5*pos_emb_2x.weight.data[i+1]
            

    # model.transformer.wpe = pos_emb_2x
    interp_len = int(model.transformer.wpe.weight.data.shape[0]*2)

    # for i in range(model.config.n_layer):
    #     model.transformer.h[i].attn.bias = torch.tril(torch.ones((interp_len, interp_len), dtype=torch.bool)).view(
    #             1, 1, interp_len, interp_len
    #         )
    pos_emb_2x = torch.nn.Embedding(interp_len, model.transformer.wpe.weight.data.shape[1], dtype=torch.bfloat16)
    pos_emb_2x.weight.data[range(0,interp_len, 2)] = model.transformer.wpe.weight.data
    
    for i in range(1, interp_len-1, 2):
        # pos_emb_2x.weight.data[i] = pos_emb_2x.weight.data[i-1]
        if i < 10:
            pos_emb_2x.weight.data[i] = pos_emb_2x.weight.data[i-1]*0.2 + pos_emb_2x.weight.data[i+1]*0.8 
        else:
            k = 7 # k has to be odd
            start_anchor = max(0, i-k)
            end_anchor = min(pos_emb_2x.weight.data.shape[0]-1, i+k)
            approx_range = [i for i in range(start_anchor+2, i, 2)] + [i for i in range(i+1, end_anchor, 2)]
            
            pos_emb_2x.weight.data[i] = interpolate_vectors(pos_emb_2x.weight.data[approx_range], 10.0)
        

    model.transformer.wpe = pos_emb_2x


def replace_attn_layer():
    for i, attn_block in enumerate(model.transformer.h):
        attn_block.attn = CustomAttn(
            model.config,
            i,
            attn_block.attn.c_attn, 
            attn_block.attn.c_proj, 
            attn_block.attn.attn_dropout, 
            attn_block.attn.resid_dropout)

def disable_position_embeddings():
    model.transformer.wpe.weight.requires_grad=False
    if rope_enabled:
        model.transformer.wpe.weight.data = torch.zeros((CTX_LEN*INTERPOLATION_FACTOR, model.transformer.wpe.weight.data.shape[1]), dtype=torch.bool)
    else:
        model.transformer.wpe.weight.data = torch.zeros((CTX_LEN, model.transformer.wpe.weight.data.shape[1]), dtype=torch.bool)


def lock_and_lower_pe(scaling_factor):
    model.transformer.wpe.weight.data = model.transformer.wpe.weight.data * scaling_factor
    model.transformer.wpe.weight.requires_grad=False

def disable_grad_except_wpe():
    for name, param in model.named_parameters():
        if name == 'transformer.wpe.weight':
            # Keep this parameter trainable
            param.requires_grad = True
        else:
            # Freeze all other parameters
            param.requires_grad = False


if __name__ == '__main__':
    from try_models import run_perplexity_measure
    from needle_in_haystack import run_needle_in_haystack_test
    
    
    if rope_enabled or alibi_enabled:
        disable_position_embeddings() # We do this here also for memory constraints.
    
    # Replace model architecture to rope/alibi and add flash attention.
    replace_attn_layer()
    
    if rope_enabled or alibi_enabled:
        disable_position_embeddings()
        ctx_len = CTX_LEN*INTERPOLATION_FACTOR
    else:
        interpolate_pos_emb() # Interpolate current PE to double ctx
        ctx_len = CTX_LEN
    model.to(device)
    model.config.n_ctx = ctx_len
    model.config.n_positions = ctx_len
    model.generation_config.max_length = ctx_len



    training_args = TrainingArguments(
        report_to="wandb" if wandb_log else None,   # enable logging to wandb
        output_dir=OUT_DIR,                         # output directory
        num_train_epochs=5,                         # total number of training epochs
        per_device_train_batch_size=2,              # batch size per device during training
        per_device_eval_batch_size=2,               # batch size for evaluation
        warmup_steps=300,                           # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                          # strength of weight decay
        lr_scheduler_type='cosine',                 # learning rate scheduler type
        learning_rate=1e-4,                         # learning rate, 1e-4 prev
        logging_steps=10,                           # log every x updates
        evaluation_strategy="steps",                # evaluate every eval_steps
        eval_steps=100,                             # evaluation steps
        gradient_accumulation_steps=2,              # gradient accumulation steps
        max_grad_norm=1.0,                          # max gradient norm
        bf16=True                                   # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=CLMDataset(torch.load(TRAIN_DATA)),
        eval_dataset=CLMDataset(torch.load(EVAL_DATA)),
        max_seq_length=ctx_len,
        dataset_text_field="text",
    )

    
    model.transformer.wte.weight.requires_grad=False # We only lock the Token Embeddings
    
    trainer.train()
    trainer.save_model(output_dir=OUT_DIR)
    # model.save_pretrained(OUT_DIR)

    doing_inference = True
    model.eval()
    run_needle_in_haystack_test(model, ctx_4k=True, ctx_8k=False)
    # run_perplexity_measure(model,ctx_4k=False)
    
