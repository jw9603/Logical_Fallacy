from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import json, evaluate
import numpy as np

import matplotlib.pyplot as plt

def load_dataset(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')
    
    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        if sample[1] == 'appeal to emotion':
            data['train']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0])
        if sample[1] == 'appeal to emotion':
            data['dev']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        if sample[1] == 'appeal to emotion':
            data['test']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data      

def load_dataset_cg(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')
    
    for sample in json_data['train']:
        data['train']['text'].append(sample[6])
        if sample[1] == 'appeal to emotion':
            data['train']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[6])
        if sample[1] == 'appeal to emotion':
            data['dev']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[6])
        if sample[1] == 'appeal to emotion':
            data['test']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data   


def load_dataset_ex(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')
    
    for sample in json_data['train']:
        # text_concat = sample[0] + ' ' + sample[7]
        # data['train']['text'].append(text_concat)
        data['train']['text'].append(sample[7])
        if sample[1] == 'appeal to emotion':
            data['train']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[7])
        if sample[1] == 'appeal to emotion':
            data['dev']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
            
            
    for sample in json_data['test']:
        
        data['test']['text'].append(sample[7])
        if sample[1] == 'appeal to emotion':
            data['test']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data   



def load_dataset_go(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')
    
    for sample in json_data['train']:
        data['train']['text'].append(sample[8])
        if sample[1] == 'appeal to emotion':
            data['train']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[8])
        if sample[1] == 'appeal to emotion':
            data['dev']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[8])
        if sample[1] == 'appeal to emotion':
            data['test']['label'].append(0)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data   








def tokenize_sequence(samples, tokenizer):
    # 'text' 필드에서 토큰화 수행
    tokenized_inputs = tokenizer(samples['text'], padding=True, truncation=True)
    return tokenized_inputs

def load_model():
    tokenizer_hf = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=5,ignore_mismatched_sizes=True)
    
    return tokenizer_hf, model

    

if __name__ =='__main__':
    
    data_path = './new_data/Argotario/argotario_all.json'
    import torch
    torch.set_printoptions(threshold=1000000000000000000)
    
    with open('./result/roberta/Argotario/Roberta_no_train_query_specific_v2_result_1time_5class.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout

        sys.stdout = output_file
        

        ###################################### Original dataset ########################################
        
        from transformers import AutoTokenizer, RobertaModel
        # 토크나이저 초기화
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        # 데이터 불러오기
        data_path = './new_data/Argotario/argotario_all.json'
        dataset = load_dataset(data_path)
        inputs=tokenizer(dataset,return_tensor='pt')
        outputs=model(**inputs)
        
        # dataset = load_dataset(data_path)
        print('dataset',dataset)
        assert -1 == 0
        # tknz, mdl = load_model()
        # tokenized_data = dataset.map(tokenize_sequence, batched=True)
        tokenized_data = dataset.map(lambda x: tokenize_sequence(x, tknz), batched=True)
        print('tokenized_data',tokenized_data)
        train_predictions = mdl(tokenized_data['train'])
        print('train_predictions',train_predictions)
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(train_predictions.logits) / np.exp(train_predictions.logits).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(train_predictions.logits, axis=-1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data['train']['label']
        predicted_labels = test_predict

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        
        # Micro F1 점수 계산
        mf1_test = f1_score(test_labels, predicted_labels, average='micro')
        print('Micro F1 score in TEST:', mf1_test)
        
        print('=================================================================================================')
        assert -1 == 0
        ###################################### CG dataset ########################################
        dataset_cg = load_dataset_cg(data_path)
        tknz, mdl = load_model()
        tokenized_data_cg = dataset_cg.map(tokenize_sequence, batched=True)
        
        train_predictions_cg = mdl(tokenized_data_cg['train'])
  
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(train_predictions_cg.logits) / np.exp(train_predictions_cg.logits).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(train_predictions_cg.logits, axis=-1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_cg['train']['label']
        predicted_labels = test_predict

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        
        # Micro F1 점수 계산
        mf1_test = f1_score(test_labels, predicted_labels, average='micro')
        print('Micro F1 score in TEST:', mf1_test)
        
        print('=================================================================================================')
  
        print('=================================================================================================')
        ###################################### EX dataset ########################################
        dataset_ex = load_dataset_ex(data_path)
        tknz, mdl = load_model()
        tokenized_data_ex = dataset_ex.map(tokenize_sequence, batched=True)
        train_predictions_ex = mdl(tokenized_data_ex['train'])
        
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(train_predictions_ex.logits) / np.exp(train_predictions_ex.logits).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(train_predictions_ex.logits, axis=-1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_ex['train']['label']
        predicted_labels = test_predict

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        
        # Micro F1 점수 계산
        mf1_test = f1_score(test_labels, predicted_labels, average='micro')
        print('Micro F1 score in TEST:', mf1_test)
        
        print('=================================================================================================')
        ###################################### GO dataset ########################################
        dataset_go = load_dataset_go(data_path)
        tknz, mdl = load_model()
        tokenized_data_go = dataset_go.map(tokenize_sequence, batched=True)
        train_predictions_go = mdl(**tokenized_data_go['train'])
        
        
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(train_predictions_go.logits) / np.exp(train_predictions_go.logits).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(train_predictions_go.logits, axis=-1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_go['train']['label']
        predicted_labels = test_predict

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        
        # Micro F1 점수 계산
        mf1_test = f1_score(test_labels, predicted_labels, average='micro')
        print('Micro F1 score in TEST:', mf1_test)
        
        print('=================================================================================================')
        
        
        # Calculate F1 scores
        def calculate_f1_scores(predictions, labels):
            preds = np.argmax(predictions.predictions, axis=1)
            return f1_score(labels, preds, average='macro')


        test_labels = dataset['test']['label']
        test_labels_cg = dataset_cg['test']['label']
        test_labels_ex = dataset_ex['test']['label']
        test_labels_go = dataset_go['test']['label']

        f1_original = calculate_f1_scores(train_predictions, test_labels)
        f1_cg = calculate_f1_scores(train_predictions_cg, test_labels_cg)
        f1_ex = calculate_f1_scores(train_predictions_ex, test_labels_ex)
        f1_go = calculate_f1_scores(train_predictions_go, test_labels_go)

        print('Original F1 Score:', f1_original)
        print('CG F1 Score:', f1_cg)
        print('EX F1 Score:', f1_ex)
        print('Go F1 Score:', f1_go)
        
        sys.stdout = original_stdout
    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")