import torch
from dataset import dataloader
from transformer import Transformer
from config import MODEL_PATH,MODEL_SIZE,MODEL_NUMBER_OF_BLOCKS,\
                   MODEL_NUMBER_OF_HEADS,MODEL_HEAD_DIM,LEARNING_RATE,\
                   EPOCHS,PRINT_FREQ,SAVE_FREQ

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"running on device: {device}")

model = Transformer(MODEL_SIZE, 
                    MODEL_NUMBER_OF_BLOCKS, 
                    MODEL_NUMBER_OF_HEADS,
                    MODEL_HEAD_DIM).to(device)

print("model created, total number of learnable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

try:
    print(f"loading checkpoint from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("loading done")

except FileNotFoundError:
    print(f"{MODEL_PATH} not found, training a new model")
    

CE = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)

for e in range(EPOCHS):

    for i,(xb,yb) in enumerate(iter(dataloader)):

        try:
            xb = xb.to(device)
            yb = yb.to(device)

            predb = model(xb)

            # predb is transposed because for some fucking reason torch 
            # always wants the classes as the second dimension
            loss = CE(predb.mT,yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % PRINT_FREQ == 0:
                print(f"iter {i} train loss {loss}")
            if i % SAVE_FREQ == 0:
                print(f"saving model to {MODEL_PATH}")
                torch.save(model.state_dict(),MODEL_PATH)
        except Exception as ex:
            print(f"not enough memory for the batch | longest seq len: {xb.shape[1]} | skipping iteration")
            del xb
            del yb
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
            continue
    
    print(f"epoch {e+1} train loss {loss}")
