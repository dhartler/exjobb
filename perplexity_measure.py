import torch
import torch.nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm


model_name = "AI-Sweden-Models/gpt-sw3-126m"
# model_name = "gpt2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encodings = torch.Tensor(torch.load("./data/books_10m_tok_4096_eval.pt"))
encodings = encodings[:4, :].reshape((1, 4*4096)).to(device)

# encodings = torch.Tensor(torch.load("./data/gpt2_set_4096_eval.pt"))
# encodings = encodings[:4, :].reshape((1, 4*4096)).to(device)

def compute_perplexity(context_windows, model):
    ppl_list = []
    for ctx_len in context_windows:
        random.seed(123)
        CE_losses = []
        print(f"Running perplexity test with ctx: {ctx_len}")
        for _ in tqdm(range(0, 300, 1)):
            pred_tok_idx = random.randint(5000, 15_000)
            if ctx_len > 5000: pred_tok_idx = random.randint(8192, 15_000)
            input_ids = encodings[:, pred_tok_idx-ctx_len:pred_tok_idx]
            target_ids = input_ids.clone()
            target_ids[:, :-1] = -100 # Do not compute loss for any other labels than last one, -100 is 'ignore label'
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                cross_entropy_loss = outputs.loss
            CE_losses.append(cross_entropy_loss)
        ppl = torch.exp(torch.stack(CE_losses).mean())
        ppl_list.append(ppl.item())
    
    return ppl_list


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(f"./126m_instruct_alibi/checkpoint-5000", torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True).to(device)
    print(compute_perplexity([40, 80, 120, 800], model))
