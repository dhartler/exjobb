import torch
import torch.nn
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from perplexity_measure import compute_perplexity


# model_name = "AI-Sweden-Models/gpt-sw3-126m-instruct"
# model_name = "./126m_rope_no_epoch"

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True).to(device)
def run_perplexity_measure(model, ctx_4k=False, ctx_8k=False):
    
    if ctx_4k:
        context_window=[16,32,128, 256, 512, 1024, 1500, 2000, 2500, 3000, 3500, 4000]
    elif ctx_8k:
        context_window=[16,32,128, 256, 512, 1024, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    else:
        context_window=[16,32,128, 256, 512, 1024, 1500, 2000]

    # context_window = [250, 500]
    ppl_ls = compute_perplexity(context_windows=context_window, model=model)
    with open('./results.txt', 'a') as file:
        file.write("Perlpexity results:\n")
        file.write(str([l for l in context_window]))
        file.write("\n")
        file.write(str(ppl_ls))
        file.close()
    
    print(ppl_ls)

def interpolate_pos_emb(model):
    interp_len = model.transformer.wpe.weight.data.shape[0]*2
    for i in range(model.config.n_head):
        model.transformer.h[i].attn.bias = torch.tril(torch.ones((interp_len, interp_len), dtype=torch.bool)).view(
                1, 1, interp_len, interp_len
            ).to("cuda:0")


    pos_emb_2x = torch.nn.Embedding(interp_len, model.transformer.wpe.weight.data.shape[1], dtype=torch.bfloat16).to("cuda:0")
    pos_emb_2x.weight.data[range(0,interp_len, 2)] = model.transformer.wpe.weight.data
    for i in range(1, interp_len-1, 2):
        pos_emb_2x.weight.data[i] = 0.5*pos_emb_2x.weight.data[i-1] + 0.5*pos_emb_2x.weight.data[i+1]

    model.transformer.wpe = pos_emb_2x

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        "./1.3b_base_mix_4k_wte_locked/checkpoint-3000",
        # f"AI-Sweden-Models/gpt-sw3-1.3b", 
        torch_dtype=torch.bfloat16, 
        ignore_mismatched_sizes=True).to("cuda:0")

    # print(model)
    run_perplexity_measure(model, ctx_4k=True, ctx_8k=False)
