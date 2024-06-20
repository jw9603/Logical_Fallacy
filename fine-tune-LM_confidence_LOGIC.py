from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import json, evaluate
import numpy as np
import pickle

import matplotlib.pyplot as plt

def save_results_to_pickle(confidence_scores, labels, predicted_labels, file_path):
    results = {
        'confidence_scores': confidence_scores,
        'labels': labels,
        'predicted_labels': predicted_labels
    }
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

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
        if sample[1] == 'faulty generalization':
            data['train']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(1)
        elif sample[1] == 'false causality':
            data['train']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['train']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['train']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['train']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['train']['label'].append(6)
        elif sample[1] == 'red herring':
            data['train']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['train']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['train']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['train']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['train']['label'].append(12)
                       
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0])
        if sample[1] == 'faulty generalization':
            data['dev']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(1)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['dev']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['dev']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['dev']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['dev']['label'].append(6)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['dev']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['dev']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['dev']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['dev']['label'].append(12)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        if sample[1] == 'faulty generalization':
            data['test']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(1)
        elif sample[1] == 'false causality':
            data['test']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['test']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['test']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['test']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['test']['label'].append(6)
        elif sample[1] == 'red herring':
            data['test']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['test']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['test']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['test']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['test']['label'].append(12)
            
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
        data['train']['text'].append(sample[0] + '[SEP]'+sample[6])
        if sample[1] == 'faulty generalization':
            data['train']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(1)
        elif sample[1] == 'false causality':
            data['train']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['train']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['train']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['train']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['train']['label'].append(6)
        elif sample[1] == 'red herring':
            data['train']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['train']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['train']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['train']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['train']['label'].append(12)
                       
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[6])
        if sample[1] == 'faulty generalization':
            data['dev']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(1)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['dev']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['dev']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['dev']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['dev']['label'].append(6)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['dev']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['dev']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['dev']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['dev']['label'].append(12)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[6])
        if sample[1] == 'faulty generalization':
            data['test']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(1)
        elif sample[1] == 'false causality':
            data['test']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['test']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['test']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['test']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['test']['label'].append(6)
        elif sample[1] == 'red herring':
            data['test']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['test']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['test']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['test']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['test']['label'].append(12)
            
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
        data['train']['text'].append(sample[0] + '[SEP]'+sample[7])
        if sample[1] == 'faulty generalization':
            data['train']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(1)
        elif sample[1] == 'false causality':
            data['train']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['train']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['train']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['train']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['train']['label'].append(6)
        elif sample[1] == 'red herring':
            data['train']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['train']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['train']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['train']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['train']['label'].append(12)
                       
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[7])
        if sample[1] == 'faulty generalization':
            data['dev']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(1)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['dev']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['dev']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['dev']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['dev']['label'].append(6)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['dev']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['dev']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['dev']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['dev']['label'].append(12)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[7])
        if sample[1] == 'faulty generalization':
            data['test']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(1)
        elif sample[1] == 'false causality':
            data['test']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['test']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['test']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['test']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['test']['label'].append(6)
        elif sample[1] == 'red herring':
            data['test']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['test']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['test']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['test']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['test']['label'].append(12)
            
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
        data['train']['text'].append(sample[0] + '[SEP]'+sample[8])
        if sample[1] == 'faulty generalization':
            data['train']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['train']['label'].append(1)
        elif sample[1] == 'false causality':
            data['train']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['train']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['train']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['train']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['train']['label'].append(6)
        elif sample[1] == 'red herring':
            data['train']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['train']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['train']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['train']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['train']['label'].append(12)
                       
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[8])
        if sample[1] == 'faulty generalization':
            data['dev']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['dev']['label'].append(1)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['dev']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['dev']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['dev']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['dev']['label'].append(6)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['dev']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['dev']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['dev']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['dev']['label'].append(12)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[8])
        if sample[1] == 'faulty generalization':
            data['test']['label'].append(0)
        elif sample[1] == 'ad hominem':
            data['test']['label'].append(1)
        elif sample[1] == 'false causality':
            data['test']['label'].append(2)
        elif sample[1] == 'ad populum':
            data['test']['label'].append(3)
        elif sample[1] == 'circular reasoning':
            data['test']['label'].append(4)
        elif sample[1] == 'appeal to emotion':
            data['test']['label'].append(5)
        elif sample[1] == 'deductive reasoning':
            data['test']['label'].append(6)
        elif sample[1] == 'red herring':
            data['test']['label'].append(7)
        elif sample[1] == 'intentional fallacy':
            data['test']['label'].append(8)
        elif sample[1] == 'false dilemma':
            data['test']['label'].append(9)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(10)
        elif sample[1] == 'fallacy of extension':
            data['test']['label'].append(11)
        elif sample[1] == 'equivocation':
            data['test']['label'].append(12)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data   








def tokenize_sequence(samples):
    
    return tknz(samples['text'],padding=True, truncation=True,max_length=512)

def load_model():
    tokenizer_hf = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=13,ignore_mismatched_sizes=True)
    
    return tokenizer_hf, model

def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1 = metric.compute(predictions=predictions, references=labels, average='micro')
    
    # Softmax 취한 값 계산
    softmax_logits = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    
    # 각 예측에서 최대 신뢰도 점수를 계산하고 그 중 최대값을 찾음
    confidence_scores = np.max(np.max(softmax_logits, axis=-1), axis=0)
    
    return {'f1': f1, 'confidence_score': confidence_scores}


def train_model(mdl, tknz, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=40,
        load_best_model_at_end=True,
        fp16=True
    )

    # Trainer 객체 생성
    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tknz,
        data_collator=DataCollatorWithPadding(tokenizer=tknz),
        compute_metrics=compute_metrics
    )

    # 훈련 수행
    trainer.train()

  
    return trainer




    

if __name__ =='__main__':
    
    data_path = './new_data/LOGIC/LOGIC_all.json'
    import torch
    torch.set_printoptions(threshold=1000000000000000000)
    
    with open('./result/roberta/LOGIC/Roberta_query_specific_v3_result_1time_13class.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout

        sys.stdout = output_file

        
     
        ###################################### Original dataset ########################################
        dataset = load_dataset(data_path)
        print('dataset',dataset)
        tknz, mdl = load_model()
        tokenized_data = dataset.map(tokenize_sequence, batched=True)
        trainer = train_model(mdl, tknz, tokenized_data)
        test_predictions = trainer.predict(tokenized_data['test'])
    
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(test_predictions.predictions) / np.exp(test_predictions.predictions).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='micro')
        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data['test']['label']
        predicted_labels = test_predictions.predictions.argmax(axis=1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        print('Micro F1 score in TEST:', mf1_test)
        save_results_to_pickle(confidence_scores,test_labels,predicted_labels,'./result/LOGIC/data/score_orig1.pkl')
        # assert -1 == 0
        print('=================================================================================================')

        ###################################### CG dataset ########################################
        dataset_cg = load_dataset_cg(data_path)
        tknz, mdl = load_model()
        tokenized_data_cg = dataset_cg.map(tokenize_sequence, batched=True)
        trainer = train_model(mdl, tknz, tokenized_data_cg)
        test_predictions_cg = trainer.predict(tokenized_data_cg['test'])
  
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(test_predictions_cg.predictions) / np.exp(test_predictions_cg.predictions).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(test_predictions_cg.predictions, axis=-1)
        
        mf1_test = f1_score(tokenized_data_cg['test']['label'], test_predict, average='micro')
        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_cg['test']['label']
        predicted_labels = test_predictions_cg.predictions.argmax(axis=1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        print('Micro F1 score in TEST:', mf1_test)

        print('Micro F1 score in TEST:', mf1_test)
        save_results_to_pickle(confidence_scores,test_labels,predicted_labels,'./result/LOGIC/data/score_cg1.pkl')
        # assert -1 == 0
  
        print('=================================================================================================')
        ###################################### EX dataset ########################################
        dataset_ex = load_dataset_ex(data_path)
        tknz, mdl = load_model()
        tokenized_data_ex = dataset_ex.map(tokenize_sequence, batched=True)
        trainer= train_model(mdl, tknz, tokenized_data_ex)
        test_predictions_ex = trainer.predict(tokenized_data_ex['test'])
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(test_predictions_ex.predictions) / np.exp(test_predictions_ex.predictions).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(test_predictions_ex.predictions, axis=-1)
        
        mf1_test = f1_score(tokenized_data_ex['test']['label'], test_predict, average='micro')
        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_ex['test']['label']
        predicted_labels = test_predictions_ex.predictions.argmax(axis=1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        print('Micro F1 score in TEST:', mf1_test)

        print('Mico F1 score in TEST:', mf1_test)
        save_results_to_pickle(confidence_scores,test_labels,predicted_labels,'./result/LOGIC/data/score_ex1.pkl')
        # assert -1 == 0
        print('===============================================================================`==================')
        ###################################### GO dataset ########################################
        dataset_go = load_dataset_go(data_path)
        tknz, mdl = load_model()
        tokenized_data_go = dataset_go.map(tokenize_sequence, batched=True)
        trainer = train_model(mdl, tknz, tokenized_data_go)
        
        test_predictions_go = trainer.predict(tokenized_data_go['test'])
        # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
        probs = np.exp(test_predictions_go.predictions) / np.exp(test_predictions_go.predictions).sum(axis=-1, keepdims=True)
        print('probs',probs)
        print(probs.shape)
        # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
        confidence_scores = np.max(probs, axis=-1)
        print('confidence_scores',confidence_scores)
        print(confidence_scores.shape)
        test_predict = np.argmax(test_predictions_go.predictions, axis=-1)

        mf1_test = f1_score(tokenized_data_go['test']['label'], test_predict, average='micro')
        # 테스트 데이터의 실제 레이블과 예측된 레이블 가져오기
        test_labels = tokenized_data_go['test']['label']
        predicted_labels = test_predictions_go.predictions.argmax(axis=1)

        # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
        print("실제 레이블:", test_labels)
        print("예측된 레이블:", predicted_labels)
        print('Micro F1 score in TEST:', mf1_test)

        print('Macro F1 score in TEST:', mf1_test)
        save_results_to_pickle(confidence_scores,test_labels,predicted_labels,'./result/LOGIC/data/score_go1.pkl')
        # assert -1 == 0
        print('=================================================================================================')
        
        
        # Calculate F1 scores
        def calculate_f1_scores(predictions, labels):
            preds = np.argmax(predictions.predictions, axis=1)
            return f1_score(labels, preds, average='macro')


        test_labels = dataset['test']['label']
        test_labels_cg = dataset_cg['test']['label']
        test_labels_ex = dataset_ex['test']['label']
        test_labels_go = dataset_go['test']['label']

        f1_original = calculate_f1_scores(test_predictions, test_labels)
        f1_cg = calculate_f1_scores(test_predictions_cg, test_labels_cg)
        f1_ex = calculate_f1_scores(test_predictions_ex, test_labels_ex)
        f1_go = calculate_f1_scores(test_predictions_go, test_labels_go)

        print('Original F1 Score:', f1_original)
        print('CG F1 Score:', f1_cg)
        print('EX F1 Score:', f1_ex)
        print('Go F1 Score:', f1_go)
        
        sys.stdout = original_stdout
    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")