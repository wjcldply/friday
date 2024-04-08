pd.set_option('mode.chained_assignment',  None) # 경고를 끈다

def predict_v2(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model1.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model1(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            return np.argmax(logits)

def isEnglishOrKorean(input_s): # 영어 가사 0으로 처리하기 위한 함수
    k_count = 0
    e_count = 0
    for c in input_s:
        if ord('가') <= ord(c) <= ord('힣'):
            k_count += 1
        elif ord('a') <= ord(c.lower()) <= ord('z'):
            e_count += 1
    return "한국어" if k_count > e_count else "영어"


# 데이터 불러오기
df = pd.read_excel('song_info_data.xlsx')

# Nan 값 -> 0으로 처리
df = df.fillna(value=0)

# 문장의 절반 이상이 영어면 0으로 처리
for i in range(0,len(df)):
  for j in range(3,len(df.columns)):

    if df.iloc[i,j] != 0:

      if isEnglishOrKorean(df.iloc[i,j]) == "영어":
        df.iloc[i,j] = 0


# 노래 가사에 대한 감정 분석 코드

sentimented_list = []     # [['곡명', '가수', '감사하는 수치', '당황한 수치', ... , '회의적인 수치'],
                          #  ['곡명', '가수', '감사하는 수치', '당황한 수치', ... , '회의적인 수치'],
                          #  ['곡명', '가수', '감사하는 수치', '당황한 수치', ... , '회의적인 수치'], ... ]

for i in tqdm(range(0,len(df))):

  tmp_list = []     # ['곡명', '가수', '감사하는 수치', '당황한 수치', ... , '회의적인 수치']

  tmp_list.append(df.loc[i][0])      # 노래 제목 추가
  tmp_list.append(df.loc[i][1])      # 노래 가수 추가

  tmp_list_2 = [0 for i in range(15)]     # [0,0, ..., 0] : 15개의 감정 횟수를 위한 리스트

  tmp_list += tmp_list_2     # ['곡명', '가수', 0, 0, ... , 0 ]

  for j in range(2,len(df.columns)-1):

    if df.loc[i][j] != 0:

      pred = predict_v2(df.loc[i][j])

      tmp_list[pred+2] += 1

  sentimented_list.append(tmp_list)

df_ = pd.DataFrame(sentimented_list)

df_.to_excel('')