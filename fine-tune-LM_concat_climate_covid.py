from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import json, evaluate
import numpy as np
import os

# 1번 GPU를 사용하도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        if sample[1] == 'cherry picking':
            data['train']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'false causality':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['train']['label'].append(5)
        elif sample[1] == 'strawman':
            data['train']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['train']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(8)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0])
        if sample[1] == 'cherry picking':
            data['dev']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['dev']['label'].append(5)
        elif sample[1] == 'strawman':
            data['dev']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['dev']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(8)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        if sample[1] == 'cherry picking':
            data['test']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'false causality':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['test']['label'].append(5)
        elif sample[1] == 'strawman':
            data['test']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['test']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(8)
            
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
        if sample[1] == 'cherry picking':
            data['train']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'false causality':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['train']['label'].append(5)
        elif sample[1] == 'strawman':
            data['train']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['train']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(8)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[6])
        if sample[1] == 'cherry picking':
            data['dev']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['dev']['label'].append(5)
        elif sample[1] == 'strawman':
            data['dev']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['dev']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(8)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[6])
        if sample[1] == 'cherry picking':
            data['test']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'false causality':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['test']['label'].append(5)
        elif sample[1] == 'strawman':
            data['test']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['test']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(8)
            
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
        if sample[1] == 'cherry picking':
            data['train']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'false causality':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['train']['label'].append(5)
        elif sample[1] == 'strawman':
            data['train']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['train']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(8)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[7])
        if sample[1] == 'cherry picking':
            data['dev']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['dev']['label'].append(5)
        elif sample[1] == 'strawman':
            data['dev']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['dev']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(8)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[7])
        if sample[1] == 'cherry picking':
            data['test']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'false causality':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['test']['label'].append(5)
        elif sample[1] == 'strawman':
            data['test']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['test']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(8)
            
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
        if sample[1] == 'cherry picking':
            data['train']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['train']['label'].append(1)
        elif sample[1] == 'red herring':
            data['train']['label'].append(2)
        elif sample[1] == 'false causality':
            data['train']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['train']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['train']['label'].append(5)
        elif sample[1] == 'strawman':
            data['train']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['train']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['train']['label'].append(8)
            
    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0] + '[SEP]'+sample[8])
        if sample[1] == 'cherry picking':
            data['dev']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['dev']['label'].append(1)
        elif sample[1] == 'red herring':
            data['dev']['label'].append(2)
        elif sample[1] == 'false causality':
            data['dev']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['dev']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['dev']['label'].append(5)
        elif sample[1] == 'strawman':
            data['dev']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['dev']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['dev']['label'].append(8)
            
            
    for sample in json_data['test']:
        data['test']['text'].append(sample[8])
        if sample[1] == 'cherry picking':
            data['test']['label'].append(0)
        elif sample[1] == 'vagueness':
            data['test']['label'].append(1)
        elif sample[1] == 'red herring':
            data['test']['label'].append(2)
        elif sample[1] == 'false causality':
            data['test']['label'].append(3)
        elif sample[1] == 'irrelevant authority':
            data['test']['label'].append(4)
        elif sample[1] == 'evading the burden of proof':
            data['test']['label'].append(5)
        elif sample[1] == 'strawman':
            data['test']['label'].append(6)
        elif sample[1] == 'false analogy':
            data['test']['label'].append(7)
        elif sample[1] == 'faulty generalization':
            data['test']['label'].append(8)
            
    final_data = DatasetDict()
    for k,v in data.items():
        final_data[k] = Dataset.from_dict(v)
    
    return final_data  








def tokenize_sequence(samples):
    
    return tknz(samples['text'],padding=True, truncation=True,max_length=512)




def load_model():
    tokenizer_hf = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=9,ignore_mismatched_sizes=True)
    
    return tokenizer_hf, model


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')
    
    
    
def train_model(mdl, tknz, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=40,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tknz,
        data_collator=DataCollatorWithPadding(tokenizer=tknz),
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


if __name__ =='__main__':
    
  
    data_path = './data/covid/covid_all.json' # Combine the training, validation, and test datasets.
    import torch
    torch.set_printoptions(threshold=1000000000000000000)
    
    with open('./result/covid_roberta_go_query_lr_1e5_ml512_ep40_bs_8_result_1time_13class.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        
        sys.stdout = output_file
        
        confidence_score_list = []
       
        
        dataset = load_dataset_go(data_path)
            
        print('dataset',dataset)
            
        shuffled_dataset = dataset.shuffle(seed=42)
            
            
        tknz, mdl = load_model()
            
            
        tokenized_data = shuffled_dataset.map(tokenize_sequence,batched=True)
            
        trainer = train_model(mdl,tknz,tokenized_data)
            
        # GENERATE PREDICTIONS FOR DEV AND TEST
        print('Original Result')
        dev_predictions = trainer.predict(tokenized_data['dev'])
        dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
        test_predictions = trainer.predict(tokenized_data['test'])

        test_predict = np.argmax(test_predictions.predictions, axis=-1)

        mf1_dev = f1_score(tokenized_data['dev']['label'], dev_predict, average='macro')
        # mf1_test = f1_score(tokenized_data['test']['label'], test_predict, average='macro')

        print('Macro F1 score in DEV:', mf1_dev)
        # from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # # Evaluate precision, recall, F1-score, and accuracy for the test set
        # precision, recall, f1, _ = precision_recall_fscore_support(tokenized_data['test']['label'], test_predict, average='macro')
        # accuracy = accuracy_score(tokenized_data['test']['label'], test_predict)

        # print('Precision in TEST:', precision)
        # print('Recall in TEST:', recall)
        # # print('Macro F1 score in TEST:', f1)
        # print('Accuracy in TEST:', accuracy)
       

        sys.stdout = original_stdout
        
    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")