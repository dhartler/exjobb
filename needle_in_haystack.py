import torch
import torch.nn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "AI-Sweden-Models/gpt-sw3-126m-instruct"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)


encodings = torch.Tensor(torch.load("./data/5_books_4000_eval.pt"))
encodings = encodings[:4].reshape((1, 4*encodings.size(1))) # [1, 16_000]
pre="""
<|endoftext|><s>
User:
"""
post="""
<s>
Bot:
"""

####### PREDEFINED QA, [ (fact, question, answer), (fact, question, answer), ... ].
qa_data = [
    (
        ". \n\nPeter har en röd tröja.\n\n ",
        ". Vilken färg har Peter's tröja? Svara bara med rätt svar.",
        "röd",
    ),
    (
        ". \n\nLeonardo har en hund som heter fido.\n\n",
        ". Vad heter Leonardo's hund? Svara bara med rätt svar.",
        "fido",
    ),
    (
        ". \n\nTim gillar att att äta mat, hans favoriträtt är sushi.\n\n",
        "Vad är Tim's favoriträtt? Svara bara med rätt svar.",
        "sushi",
    ),
    (
        ". \n\nDaniel är en av världens mest kända författare. Han föddes år 1824 och är väldigt gammal.\n\n",
        "Vilket år föddes Daniel? Svara bara med rätt svar.",
        "1824",
    ),
    (
        ". \n\nAnna älskar att läsa böcker, speciellt romaner.\n\n",
        ". Vilken typ av böcker älskar Anna att läsa? Svara bara med rätt svar.",
        "romaner",
    ),
    (
        ". \n\nMax har en katt som heter Simba.\n\n",
        ". Vad heter Max's katt? Svara bara med rätt svar.",
        "simba",
    ),
    (
        ". \n\nPer är allergisk mot jordnötter.\n\n",
        "Vad är Per allergisk mot? Svara bara med rätt svar.",
        "jordnötter",
    ),
    (
        ". \n\nEmma bor i Skåne.\n\n",
        "Var bor Emma? Svara bara med rätt svar.",
        "skåne",
    ),
    (
        ". \n\nOlof är född och uppvuxen i Stockholm.\n\n",
        "Var är Olof född och uppvuxen? Svara bara med rätt svar.",
        "stockholm",
    ),
    (
        ". \n\nViktor är en professionell kock och arbetar på en lyxrestaurang.\n\n",
        "Vad arbetar Viktor som? Svara bara med rätt svar.",
        "kock",
    ),
    (
        ". \n\nKarin spelar fiol i en orkester.\n\n",
        "Vilket instrument spelar Karin? Svara bara med rätt svar.",
        "fiol",
    ),
    (
        ". \n\nNina tränar ofta yoga för att hålla sig i form.\n\n",
        "Vad tränar Nina ofta? Svara bara med rätt svar.",
        "yoga",
    )
]
# 13 QA


pre = tokenizer(pre, return_tensors="pt")["input_ids"]      
post = tokenizer(post, return_tensors="pt")["input_ids"]
qa_data = [( tokenizer(e[0], return_tensors="pt")["input_ids"], tokenizer(e[1], return_tensors="pt")["input_ids"], e[2]) for e in qa_data]
min_ctx_len = max([e[0].shape[1] + e[1].shape[1] for e in qa_data])


def needle_in_a_haystack(ctx_len:int, p:float, model, q_num=1):
    if ctx_len <= min_ctx_len:
        raise ValueError("""Context length must be at least more than {min_ctx_len} tokens for needle in a haystack test""")
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be a number between 0.0 and 1.0") 
    
    num_filler_tokens = ctx_len - min_ctx_len
    filler_tokens = encodings[:, 0:num_filler_tokens]
    pre_filler = filler_tokens[0,  0:int((1-p)*num_filler_tokens)].reshape(1,-1)
    post_filler = filler_tokens[0,  int((1-p)*num_filler_tokens):].reshape(1,-1)

    ans = qa_data[q_num][2]
    ctx_window = torch.hstack([pre, pre_filler, qa_data[q_num][0], post_filler, qa_data[q_num][1], post]).to(device)

    with torch.no_grad():
        generated_token_ids = model.generate(
        inputs=ctx_window,
        max_new_tokens=15,
        do_sample=True,
        temperature=0.1,
        top_p=1,
        output_attentions =False,
        )[0]
    print(tokenizer.decode(generated_token_ids[ctx_window.shape[1]:]))
    if ans in tokenizer.decode(generated_token_ids[ctx_window.shape[1]:]).lower():
        return 1
    else:
        return 0
    
    
def run_needle_in_haystack_test(model, ctx_4k=False, ctx_8k=False):
    # Parameters for context window and hidden facts
    if ctx_4k:
        ctx_lengths=[256, 512, 1024, 1500, 2000, 2500, 3000, 3500, 3750, 4000]
    elif ctx_8k:
        ctx_lengths=[256, 512, 1024, 1500, 2000, 2500, 3000, 3500, 3750, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    else:
        ctx_lengths=[256, 512, 1024, 1500, 2000]    
    
    fact_offsets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Compute values
    res = torch.zeros((len(ctx_lengths), len(fact_offsets)))
    for i, ctx_l in enumerate(ctx_lengths):
        for j, hidden_p in enumerate(fact_offsets):
            for question_num in range(len(qa_data)):
                res[i, j] = res[i, j] + needle_in_a_haystack(ctx_l, hidden_p, model, q_num=question_num)
                # print( needle_in_a_haystack(ctx_l, hidden_p,model, q_num=question_num) )
    with open('./results.txt', 'a') as file:
        file.write("Needle in a haystack result:\n")
        file.write(str(res / len(qa_data)))
        file.close()
    print(str(res / len(qa_data)))

if __name__ == '__main__':
    # m_type = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
    m_type = "./1.3b_inst_mix_4_wte_locked/checkpoint-1500"
    model = AutoModelForCausalLM.from_pretrained(m_type, 
        torch_dtype=torch.float16,
        ).to(device)
    run_needle_in_haystack_test(model, ctx_4k=True, ctx_8k=False)


