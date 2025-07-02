import torch
from transformer import Transformer
from config import *
from tokenizer import tokenizer

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

# always run inference on cpu so the gpu can keep training
device = torch.device('cpu') 

print(f"running on device: {device}")

model = Transformer(MODEL_SIZE, 
                    MODEL_NUMBER_OF_BLOCKS, 
                    MODEL_NUMBER_OF_HEADS,
                    MODEL_HEAD_DIM).to(device)

print("model created, total number of learnable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

softmax = torch.nn.Softmax(dim = -1)

model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
print(f"checkpoint loaded from: {MODEL_PATH}")

print()
print(bcolors.YELLOW, end = "")
print("yellow: user prompt",end = "")
print(bcolors.ENDC)
print("default: model output")
print()
print(bcolors.YELLOW, end = "")
print(START_PROMPT,end = "")
print(bcolors.ENDC, end = "")

start_prompt = tokenizer.encode("<SOS>"+START_PROMPT).ids

x = torch.Tensor([start_prompt]).to(device).to(torch.long) 

with torch.no_grad():

    for _ in range(MAX_TOKENS):
    
        if OUTPUT_TYPE == "sample":
            # sample from the probability distribution predicted
            next_token_id = torch.multinomial(softmax(model(x))[-1][-1], num_samples=1).squeeze(0)
        elif OUTPUT_TYPE == "argmax":
            # choose the token with the highest probability
            next_token_id = softmax(model(x))[-1][-1].argmax()
        x = torch.cat((x,next_token_id.unsqueeze(0).unsqueeze(0)),dim=-1)
        token = tokenizer.decode([next_token_id.item()])
    
        print(token, end="",flush=True)
        if next_token_id.item() == tokenizer.encode("<EOS>").ids[0]:
            break

print()

