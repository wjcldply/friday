# command to run : CUDA_VISIBLE_DEVICES=1 python koGPT2.py
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
import os

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

torch.cuda.set_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'device : {device}')
# print(f'Count of using GPUs : {torch.cuda.device_count()}')
# print(f'Current cuda device : {torch.cuda.current_device()}')

key = "reduced_merged"
batch_size = 64
lr_factor = 3
epoch = 5

save_dir = f'/workspace/models'
model_save_path = os.path.join(save_dir, f'koGPT2_{key}_model_batch{batch_size}_lr{lr_factor}_epoch{epoch}.pth')

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)



def chat(sentiment_label, user_input):
    try:
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', state_dict=torch.load(model_save_path)).to(device)
    except FileNotFoundError:
        # print(f'|{key}|batch{batch_size}|lr{lr_factor}|epoch{epoch}| Model Not Found in the Directory')
        return
    # print(f'|{key}|batch{batch_size}|lr{lr_factor}|epoch{epoch}| Model loaded')      
    q = user_input
    q = re.sub(r"([?.!,])", r" ", q)
    # sentiment_label = sentiment_label_list[q_idx]  # Modified
    a = ""
    while 1:
        # Modified: sentiment token을 기본값인 '0' 대신 각 데셋 및 문장에 맞는 값으로 지정
        input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + str(sentiment_label) + A_TKN + a)).unsqueeze(dim=0).to(device)
        q_len = len(koGPT2_TOKENIZER.tokenize(Q_TKN + q + SENT + str(sentiment_label)))
        a_len = len(koGPT2_TOKENIZER.tokenize(A_TKN + a))
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, :q_len + a_len] = 1
        pred = model(input_ids, attention_mask=attention_mask)
        pred = pred.logits
        gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace("▁", " ")
        a = re.sub(r"([?.!,])", r" ", a)
    return a
