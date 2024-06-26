# python koBERT.py

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

# ★ Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import os

# ★
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

"""# **모델 파라미터 및 기본 환경 구축**"""

# Setting parameters
max_len = 64
batch_size = 192
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 2e-5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=15,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


"""# **새로운 문장 테스트 해보기**"""

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids).to(device)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("감사하는")
            elif np.argmax(logits) == 1:
                test_eval.append("당황한")
            elif np.argmax(logits) == 2:
                test_eval.append("분노한")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픈")
            elif np.argmax(logits) == 4:
                test_eval.append("불안한")
            elif np.argmax(logits) == 5:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 6:
                test_eval.append("외로운")
            elif np.argmax(logits) == 7:
                test_eval.append("편안한")
            elif np.argmax(logits) == 8:
                test_eval.append("스트레스 받는")
            elif np.argmax(logits) == 9:
                test_eval.append("부끄러운")
            elif np.argmax(logits) == 10:
                test_eval.append("실망한")
            elif np.argmax(logits) == 11:
                test_eval.append("열등감을 느끼는")
            elif np.argmax(logits) == 12:
                test_eval.append("후회되는")
            elif np.argmax(logits) == 13:
                test_eval.append("자신하는")
            elif np.argmax(logits) == 14:
                test_eval.append("회의적인")
    
            return (test_eval[0])


"""# **학습된 모델 불러오기**"""

model = torch.load('/workspace/models/model_koBERT_batch64_lr3e-05_epoch10.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load('/workspace/models/state_koBERT_batch64_lr3e-05_epoch10.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load('/workspace/models/all_koBERT_batch64_lr3e-05_epoch10.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])

model.to(device)


"""
test_questions = [
    "남들 보는 앞에서 교수님한테 엄청 깨졌어.",
    "하루 종일 일에 쫓기는 기분이야.",
    "친구랑 싸워서 아직도 분이 안 풀려.",
    "내 계획은 이게 아니었는데...",
    "이번 발표는 준비를 제대로 못 해서 걱정이다.",
    "그래도 생각보다는 성적이 괜찮게 나왔어.",
    "헤어진 전 애인이 보고싶다.",
    "부모님께 드릴 선물 사러 가는 길인데 신나.",
    "애인에게 받은 선물이 너무 맘에 들어."
]

for test_q in test_questions:
    print(f'| User({test_q}) | Predicted Sentiment({predict(test_q)})|')
    """

"""#질문 무한반복하기! 0 입력시 종료
end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0" :
      break
    predict(sentence)
    print("\n")
"""