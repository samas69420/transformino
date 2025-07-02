from torch.utils.data import Dataset,DataLoader
from tokenizer import tokenizer
import torch
from config import CSV_DATASET_PATH, BATCH_SIZE
import pandas as pd


def collate_fn(batch):

    batch_x = [e[0] for e in batch]
    batch_y = [e[1] for e in batch]

    maxlen = max([len(e) for e in batch_x])
    padded_batch_x = []
    for e in batch_x:
        padded_batch_x.append(e+[tokenizer.encode("<PAD>").ids[0]]*(maxlen-len(e)))

    maxlen = max([len(e) for e in batch_y])
    padded_batch_y = []
    for e in batch_y:
        padded_batch_y.append(e+[tokenizer.encode("<PAD>").ids[0]]*(maxlen-len(e)))

    x_batched,y_batched = torch.tensor(padded_batch_x),torch.tensor(padded_batch_y)
    
    return x_batched,y_batched


class TextDataset(Dataset):


    def __init__(self, csv_path):

        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]
        text = row.text
        text = "<SOS>" + str(text) + "<EOS>"

        tokenized = tokenizer.encode(text).ids

        x = tokenized[:-1]
        y = tokenized[1:]

        # x,y here are only lists, the conversion in tensors is in collate_fn
        return x,y 


    def __len__(self):
        return self.len

text_dataset = TextDataset(CSV_DATASET_PATH)
dataloader = DataLoader(dataset = text_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    print("len of dataset:",len(text_dataset))
    e = next(iter(dataloader))
    print("shape of batched input:",e[0].shape)
    print(e[0])

# visual representation of data from the dataset,
# shape of batched target tensor is the same
#  
#                   _              _  
#                  |    id(word1)   |    
#                  |   _  ...       |_   
#   batched        |  |   id(word1)   | \
# input tensor     |  |   id(word2)   | |
#                  |  |   id(word3)   | | padded seq len (max len in batch)
#                _ |_ |     ...       | |
#     batch size \    |    id(pad)    | |
#                 \_  |_   id(pad)   _| /   
#                                          shape = (batch,seq_len)
#                               
#
# visual representation of data afer embedding 
# (this will happen inside the model)
#                  _                 _  
#                 |     emb(word1)    |    
#                 |   _     ...       | _   
#  batched        |  |     emb(word1)    | \
#input tensor     |  |     emb(word2)    | |
#                 |  |     emb(word3)    | | padded seq len
#               _ |_ |        ...        | |
#    batch size \    |      emb(pad)     | |
#                \_  |_     emb(pad)    _| /
#
#                     \_________________/
#                        embedding size
#                                          shape = (batch,seq_len,embedding size)    
