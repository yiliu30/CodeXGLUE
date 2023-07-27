# author Yi 
# 2023/7
# fine tuning the codebert on code-detection

import json

def read_data(file_path):
    texts = []
    labels = []
    with open(file_path) as f:
        for line in f:
            js=json.loads(line.strip())
            code = ' '.join(js['func'].split())
            texts.append(code)
            labels.append(js['target'])
    
    return texts, labels

train_texts, train_labels = read_data("./dataset/train.jsonl")
val_texts, val_labels = read_data("./dataset/valid.jsonl")
# import pdb; pdb.set_trace()

# from transformers import DistilBertTokenizerFast
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# model_name_or_path = "/home/st_liu/workspace/inc_examples/mrm8488/codebert-base-finetuned-detect-insecure-code"
model_name_or_path = "/home/st_liu/workspace/inc_examples/microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
train_encodings = tokenizer(train_texts, return_tensors="pt", truncation=True, padding='max_length')
val_encodings = tokenizer(val_texts, return_tensors="pt", truncation=True, padding='max_length')

import torch

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CodeDataset(train_encodings, train_labels)
val_dataset = CodeDataset(val_encodings, val_labels)


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification, AdamW

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
# model.to(device)
# model.train()

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

import numpy as np


def evaluate(model):
    print("***　eval model .. ")
    all_labels = []
    all_preds = []
    for idx, batch in enumerate(val_loader):
        # if idx > 3:
        #     break
        # import pdb; pdb.set_trace()
        model.eval()
        with torch.no_grad():
            labels = batch.pop('labels')
            inputs = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            # print(np.argmax(logits.detach().numpy(), axis=1))
            all_labels.append(labels.numpy())
            all_preds.append(np.argmax(logits.detach().numpy(), axis=1))
            np.concatenate(all_labels, axis=0)
            np.concatenate(all_preds, axis=0)
            cur_acc = np.mean(np.concatenate(all_labels, axis=0)==np.concatenate(all_preds, axis=0))
            print( "eval *** cur_acc", cur_acc)
            
            # with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
            #     for example,pred in zip(eval_dataset.examples,preds):
            #         if pred:
            #             f.write(example.idx+'\t1\n')
            #         else:
            #             f.write(example.idx+'\t0\n')    
    cur_acc = np.mean(np.concatenate(all_labels, axis=0)==np.concatenate(all_preds, axis=0))
    print("all acc", cur_acc)
    return cur_acc

from transformers import AdamW
optim = AdamW(model.parameters(), lr=5e-5)
device = 'cpu'

import logging
logger = logging.getLogger(__name__)
import os
results = {'eval_acc': 0}

# import pdb;
# pdb.set_trace()
# model.load_state_dict(torch.load("./checkpoint-best-acc-new/model.bin"))

# cur_acc = evaluate(model)

global_step = 0
for epoch in range(3):
    all_labels = []
    all_preds = []
    for idx , batch in enumerate(train_loader):
        
        global_step += 1
        optim.zero_grad()
        labels = batch.pop('labels')
        inputs = batch
        model.train()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        # print(np.argmax(logits.detach().numpy(), axis=1))
        all_labels.append(labels.numpy())
        all_preds.append(np.argmax(logits.detach().numpy(), axis=1))
        np.concatenate(all_labels, axis=0)
        np.concatenate(all_preds, axis=0)
        cur_acc = np.mean(np.concatenate(all_labels, axis=0)==np.concatenate(all_preds, axis=0))
        print( "cur_acc", cur_acc)
        loss.backward()
        print(f"loss: {loss.item()}")
        optim.step()
        
        
        if global_step % 100 == 0:
            best_acc=results['eval_acc']
            cur_acc = evaluate(model)
            if cur_acc > best_acc:
                results['eval_acc'] = cur_acc
                best_acc = results['eval_acc']
                print("  "+"*"*20)  
                print("  Best acc:%s",round(best_acc,4))
                print("  "+"*"*20)                          
                
                checkpoint_prefix = 'checkpoint-best-acc-new'
                output_dir = os.path.join('{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                model.config.to_json_file('config-{}-{}-{}'.format(checkpoint_prefix, epoch, idx))
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                print("Saving model checkpoint to %s", output_dir)
            

# # import pdb; 
# # pdb.set_trace()