import streamlit as st
import time
from koGPT2 import chat


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd

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


model = torch.load('/workspace/models/model_koBERT_batch64_lr3e-05_epoch10.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load('/workspace/models/state_koBERT_batch64_lr3e-05_epoch10.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load('/workspace/models/all_koBERT_batch64_lr3e-05_epoch10.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])

model.to(device)


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.emotions = []

def chatbot(prompt):
    sentiment_label = predict(prompt)
    assistant_response = chat(sentiment_label, prompt)
    return assistant_response, sentiment_label

def vectorize_emotions(emotions_list):
    emotion_vector = [0] * 15
    for emotion in emotions_list:
        if emotion == '감사하는':
            emotion_vector[0] += 1
        elif emotion == '당황한':
            emotion_vector[1] += 1
        elif emotion == '분노한':
            emotion_vector[2] += 1
        elif emotion == '슬픈':
            emotion_vector[3] += 1
        elif emotion == '불안한':
            emotion_vector[4] += 1
        elif emotion == '기쁨':
            emotion_vector[5] += 1
        elif emotion == '외로운':
            emotion_vector[6] += 1
        elif emotion == '편안한':
            emotion_vector[7] += 1
        elif emotion == '스트레스 받는':
            emotion_vector[8] += 1
        elif emotion == '부끄러운':
            emotion_vector[9] += 1
        elif emotion == '실망한':
            emotion_vector[10] += 1
        elif emotion == '열등감을 느끼는':
            emotion_vector[11] += 1
        elif emotion == '후회되는':
            emotion_vector[12] += 1
        elif emotion == '자신하는':
            emotion_vector[13] += 1
        elif emotion == '회의적인':
            emotion_vector[14] += 1
    return emotion_vector

def recommend_song(lyrics_df, user_sentiment):

  cosine_list = []

  user = np.array(user_sentiment)

  for i in range(0,len(lyrics_df)):
    music = np.array(lyrics_df.iloc[i][3:])

    dot_product = np.dot(user, music)

    magnitude1 = np.linalg.norm(user)
    magnitude2 = np.linalg.norm(music)

    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    cosine_list.append(cosine_similarity)

  max_value = max(cosine_list)
  max_index = cosine_list.index(max_value)

  return lyrics_df.iloc[max_index][1:3]

lyrics = pd.read_excel("/workspace/models/lyrics_emotions_all.xlsx")

with st.sidebar:
    st.title("💬FRIDAY")
    st.sidebar.button('Clear Chat HIstory', on_click = clear_chat_history)
    st.subheader('📖 Check out our [Github Repository](https://github.com/wjcldply/friday)!')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.emotions = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("어떤 일이 있었나요?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if prompt!='end':  # 유저가 end가 아닌 텍스트를 입력으로 주면 대화를 이어가는 것으로 간주
            assistant_response, sentiment_label = chatbot(prompt)  # 챗봇의 답변과 감정 도출하는 함수 - chatbot(prompt)
            full_response += f"입력하신 문장에서 '{sentiment_label}' 감정이 느껴져요.\n"            
            st.session_state.emotions.append(sentiment_label)  # 유저 입력 문장의 감정분석 결과 저장
        else:  # 유저가 end를 입력으로 주면 대화 종료로 간주
            # 대화가 종료되면 일어나는 작업들
            emotion_vector = vectorize_emotions(st.session_state.emotions)
            recommendation = recommend_song(lyrics, emotion_vector)
            assistant_response = f"대화에서 감지된 감정들을 기반으로 한 오늘의 추천곡은 '{recommendation[1]}'의 '{recommendation[0]}'입니다."
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})    
