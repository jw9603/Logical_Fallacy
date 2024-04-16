import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

import pickle

def save_results_to_pickle(confidence_scores, labels, predicted_labels, file_path):
    results = {
        'confidence_scores': confidence_scores,
        'labels': labels,
        'predicted_labels': predicted_labels
    }
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
def load_and_tokenize_data(file_path, tokenizer, max_length):
    with open(file_path) as f:
        data = json.load(f)
    
    label_map = {
        "cherry picking": 0,
        "vagueness": 0,
        "red herring": 0,
        "false causality": 0,
        "irrelevant authority": 0,
        "evading the burden of proof":0,
        "strawman":0,
        "false analogy":0,
        "faulty generalization":0,
        "no fallacy":1
    }
    
    inputs = []
    labels = []
    
    for sample in data['test']:
        input_text = sample[0]  # 입력 텍스트
        label = sample[1]       # 레이블
        
        # 레이블을 숫자로 매핑
        label_id = label_map[label]
        
        # 텍스트 토큰화 및 패딩
        tokenized_input = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        
        inputs.append(tokenized_input)
        labels.append(label_id)
    
    return inputs, labels


def load_and_tokenize_data_cg(file_path, tokenizer, max_length):
    with open(file_path) as f:
        data = json.load(f)
    
    label_map = {
        "cherry picking": 0,
        "vagueness": 0,
        "red herring": 0,
        "false causality": 0,
        "irrelevant authority": 0,
        "evading the burden of proof":0,
        "strawman":0,
        "false analogy":0,
        "faulty generalization":0,
        "no fallacy":1
    }
    
    inputs = []
    labels = []
    
    for sample in data['test']:
        input_text = sample[0] + '[SEP]' + sample[6]  # 입력 텍스트
        label = sample[1]       # 레이블
        
        # 레이블을 숫자로 매핑
        label_id = label_map[label]
        
        # 텍스트 토큰화 및 패딩
        tokenized_input = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        
        inputs.append(tokenized_input)
        labels.append(label_id)
    
    return inputs, labels


def load_and_tokenize_data_ex(file_path, tokenizer, max_length):
    with open(file_path) as f:
        data = json.load(f)
    
    label_map = {
        "cherry picking": 0,
        "vagueness": 0,
        "red herring": 0,
        "false causality": 0,
        "irrelevant authority": 0,
        "evading the burden of proof":0,
        "strawman":0,
        "false analogy":0,
        "faulty generalization":0,
        "no fallacy":1
    }
    
    inputs = []
    labels = []
    
    for sample in data['test']:
        input_text = sample[0] + '[SEP]' + sample[7]  # 입력 텍스트
        label = sample[1]       # 레이블
        
        # 레이블을 숫자로 매핑
        label_id = label_map[label]
        
        # 텍스트 토큰화 및 패딩
        tokenized_input = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        
        inputs.append(tokenized_input)
        labels.append(label_id)
    
    return inputs, labels


def load_and_tokenize_data_go(file_path, tokenizer, max_length):
    with open(file_path) as f:
        data = json.load(f)
    
    label_map = {
        "cherry picking": 0,
        "vagueness": 0,
        "red herring": 0,
        "false causality": 0,
        "irrelevant authority": 0,
        "evading the burden of proof":0,
        "strawman":0,
        "false analogy":0,
        "faulty generalization":0,
        "no fallacy":1
    }
    
    inputs = []
    labels = []
    
    for sample in data['test']:
        input_text = sample[0] + '[SEP]' + sample[8]  # 입력 텍스트
        label = sample[1]       # 레이블
        
        # 레이블을 숫자로 매핑
        label_id = label_map[label]
        
        # 텍스트 토큰화 및 패딩
        tokenized_input = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        
        inputs.append(tokenized_input)
        labels.append(label_id)
    
    return inputs, labels

def main():
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # 데이터 로드 및 토큰화
    file_path = './new_data/COVID-19/covid_all_with_no_fallacy.json'
    # file_path = './new_data/CLIMATE/climate_all_with_no_fallacy.json'
    max_length = 512  # 최대 길이 지정
    
    # 파일에 출력할 객체 생성
    with open('./result/roberta/COVID-19/Roberta_query_specific_no_train_binary_result_1time_9class.txt','w') as output_file:
        # 첫 번째 데이터 버전에 대한 결과 출력
        print("첫 번째 데이터 버전 결과:", file=output_file)
        inputs_0, labels_0 = load_and_tokenize_data(file_path, tokenizer, max_length)
        outputs_0 = get_model_output(inputs_0)
        confidence_scores_0,predicted_labels_0 = print_results(labels_0, outputs_0, output_file=output_file)
        save_results_to_pickle(confidence_scores_0,labels_0,predicted_labels_0,'./result/roberta/COVID/new_data/score_orig.pkl')
        # 두 번째 데이터 버전에 대한 결과 출력
        print("\n두 번째 데이터 버전 결과:", file=output_file)
        inputs_7, labels_7 = load_and_tokenize_data_cg(file_path, tokenizer, max_length)
        outputs_7 = get_model_output(inputs_7)
        confidence_scores_7, predicted_labels_7 = print_results(labels_7, outputs_7, output_file=output_file)
        save_results_to_pickle(confidence_scores_7,labels_7,predicted_labels_7,'./result/roberta/COVID/new_data/score_cg.pkl')
        # 세 번째 데이터 버전에 대한 결과 출력
        print("\n세 번째 데이터 버전 결과:", file=output_file)
        inputs_8, labels_8 = load_and_tokenize_data_ex(file_path, tokenizer, max_length)
        outputs_8 = get_model_output(inputs_8)
        confidence_scores_8,predicted_labels_8=print_results(labels_8, outputs_8, output_file=output_file)
        save_results_to_pickle(confidence_scores_8,labels_8,predicted_labels_8,'./result/roberta/COVID/new_data/score_ex.pkl')

        # 네 번째 데이터 버전에 대한 결과 출력
        print("\n네 번째 데이터 버전 결과:", file=output_file)
        inputs_9, labels_9 = load_and_tokenize_data_go(file_path, tokenizer, max_length)
        outputs_9 = get_model_output(inputs_9)
        confidence_scores_9, predicted_labels_9 = print_results(labels_9, outputs_9, output_file=output_file)
        save_results_to_pickle(confidence_scores_9,labels_9,predicted_labels_9,'./result/roberta/COVID/new_data/score_go.pkl')

def get_model_output(inputs):
    # 모델 초기화
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # 모델 입력 준비
    input_ids = torch.cat([input_dict["input_ids"] for input_dict in inputs], dim=0)
    attention_masks = torch.cat([input_dict["attention_mask"] for input_dict in inputs], dim=0)

    # 모델에 입력 데이터 전달하여 예측 수행
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    return outputs

def print_results(labels, outputs,output_file=None):
    # 소프트맥스 함수를 사용하여 로그 확률값을 확률값으로 변환
    probs = np.exp(outputs.logits) / np.exp(outputs.logits).sum(axis=-1, keepdims=True)

    # 각 샘플에 대해 최대 확률값을 계산하여 confidence score로 사용
    confidence_scores, _ = torch.max(torch.tensor(probs), dim=-1)

    # 예측된 레이블 가져오기
    predicted_labels = np.argmax(outputs.logits, axis=1)

    # 테스트 데이터의 실제 레이블과 예측된 레이블 출력
    print('confidence_scores',confidence_scores)
    print(confidence_scores.shape)
    print("실제 레이블:", labels)
    print("예측된 레이블:", predicted_labels)
    # Micro F1 스코어 계산
    mf1 = f1_score(labels, predicted_labels, average='micro')
    print("Micro F1 Score:", mf1)
    
    acc = accuracy_score(labels,predicted_labels)
    print('Acccuracy:',acc)
    
    return confidence_scores, predicted_labels

if __name__ == "__main__":
    main()
    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")