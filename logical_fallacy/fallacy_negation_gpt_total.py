import random
from openai import OpenAI
import json
from retry import retry
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
random.seed(0)

@retry()
def create_fallacy_classification_no_question_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def create_fallacy_classification_question_zero_list(text,query):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text(or QA-pair) based on the Query."},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def create_fallacy_classification_negation_zero_list(text,query,negation_text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. Negation text is when a negation is taken from the original text. The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text(or QA-pair) base on the Query.\n"},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role": "user", "content": negation_text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def create_fallacy_classification_negation1_zero_list(text,negation_text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. Negation text reverses the verb state of the original text. If there's a 'not' around the verb of the original text, remove the 'not', and if it's positive, add the negation word 'not'. The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nReferring to negation text can be helpful when detecting logical errors in the original text.\nPlease detect a fallacy in the original text.\n"},
            {"role": "user", "content": text},
            {"role": "user", "content": negation_text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def create_fallacy_classification_question_path_zero_list(text,query,reasoning_path):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. Negation text is when a negation is taken from the original text.The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text(or QA-pair) based on the Query.\n"},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role":"user","content":reasoning_path},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def create_fallacy_classification_path_zero_list(text,reasoning_path):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The text consists of question and answer pairs or sentences. The label can be 'Faulty Generalization' and 'False Causality' and 'Irrelevant Authority'.\nReasoning path consists of relations and entities separated by arrow(->).\n Relations consist of 'antonym', 'atlocation','capableof','causes','createdby','isa','desires','hassubevent','partof','hascontext','hasproperty','madeof','notcapableof','notdesires','receivesaction','relatedto' and 'usedfor'.\nplease refer to the reasoning path to detect logical fallacies in the text(or QA-pair).\n"},
            {"role": "user", "content": text},
            {"role":"user","content":reasoning_path},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode
       

if __name__ =='__main__':
    # CALLS = 0
    # FREE_TIER_LIMIT = 100  # Set this according to the free tier's rate limit
    # WAIT_TIME = 60  # Set the waiting time according to the free tier's reset time (in seconds)
    CALLS = 0  # Initialize the API call counter
    mode = 'None'
    
    client = OpenAI(
    api_key="YOURKEY"
    )
    
    # Evaluation vectors
    ground_truth = []
    gpt_preds = []
    
    with open('./data/new_total_split_v2_modified_negation_sum.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/negation/general_negation_only_prompt1_result_zero_seed0_1time_3class.txt','w') as output_file:
        import sys
        # 기존의 stdout을 백업합니다.
        original_stdout = sys.stdout
        # 출력을 파일로 리디렉션합니다.
        sys.stdout = output_file
        for sample in json_data['test']:
            # ##########################################################
            # # Check and wait if rate limit is reached
            # if CALLS >= FREE_TIER_LIMIT:
            #     print("Rate limit reached. Waiting to reset...")
            #     time.sleep(WAIT_TIME)
            #     CALLS = 0  # Reset the call counter after the wait
            #########################################################
            # Text 
            # values = [value for value in sample[-1].values() if value]
            t1 = 'Original Text: '+sample[0]
            # t2 = 'Query:' +sample[5]
            t3 = 'Negation Text:'+sample[4]
            # t3 = "Evidence Paths: " +str(values)
          
            # assert -1 == 0

            
            # Label
            if 'faulty generalization' == sample[1]:
                ground_truth.append(1)
            elif 'false causality' == sample[1]:
                ground_truth.append(2)
            elif 'irrelevant authority' == sample[1]:
                ground_truth.append(3)
            # elif 'no fallacy' == sample[1]:
                # ground_truth.append(4)
            # else:
            #     ground_truth.append(0)
                
            # Create a completion and a prediction with CHAT-CPT
           
            # completion, mode = create_fallacy_classification_no_question_zero_list(t1)
            # completion, mode = create_fallacy_classification_negation_zero_list(t1,t2,t3)
            completion, mode = create_fallacy_classification_negation1_zero_list(t1,t3)
            # completion, mode = create_fallacy_classification_path_zero_list(t1,t3)
            pred = completion.choices[0].message.content
            
            if 'generalization' in pred.lower():
                gpt_preds.append(1)
            elif 'cause' in pred.lower() or 'causality' in pred.lower():
                gpt_preds.append(2)
                
            elif 'authority' in pred.lower():
                gpt_preds.append(3)
            # elif 'no fallacy' in pred.lower() or 'not a fallacy' in pred.lower():
                # gpt_preds.append(4)
            else:
                gpt_preds.append(0)
                    
        

            CALLS += 1
            # print(f"{CALLS}/{TOTAL_CALLS} API Calls Made")
            print(CALLS,'/',TOTAL_CALLS)
            print('pred',pred)   
            # 출력 "usage" 부분
            usage = completion.usage
            print('Usage:', usage)  
        
        # Overall Accuracy
        print('ground_truth',ground_truth)
        print('ground_truth길이',len(ground_truth))
        print('gpt_preds',gpt_preds)
        print('gpt_preds길이',len(gpt_preds))
        total_accuracy = accuracy_score(ground_truth, gpt_preds)
        print("Total Accuracy:", total_accuracy)
        
    
        # Precision, Recall, F1-Score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, gpt_preds, average='macro')
        print("Precision (Macro):", precision)
        print("Recall (Macro):", recall)
        print("F1-Score (Macro):", f1)
        
        # Confusion Matrix
        cm = confusion_matrix(ground_truth, gpt_preds)
        print("Confusion Matrix:")
        print(cm)
        
        
        # 계산된 정확도를 리스트에 추가
        accuracies = []
        for label in [1, 2,3]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # 클래스별 정확도 출력
        print("Class 1 (Faulty Generalization) Accuracy:", accuracies[0])
        print("Class 2 (False Causality) Accuracy:", accuracies[1])
        print("Class 3 (Irrelevant Authority) Accuracy:", accuracies[2])
        # print("Class 4 (No Fallacy) Accuracy:", accuracies[3])
        
        
        # 클래스별로 Precision, Recall, F1-Score 계산
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)
        # 클래스별 결과 출력
        classes = ["Faulty Generalization","False Causality","Irrelevant Authority"]
        classes1 = ['The Other',"Faulty Generalization","False Causality","Irrelevant Authority"]
        if 0 in gpt_preds:
            classes = classes1
        else:
            classes = classes
        for i, class_name in enumerate(classes):
            print(f"Class {i} ({class_name}) Metrics:")
            print(f"  Precision: {class_metrics[0][i]}")
            print(f"  Recall: {class_metrics[1][i]}")
            print(f"  F1-Score: {class_metrics[2][i]}")
            print()
        
        
        

        # 파일로 리디렉션된 출력을 다시 기존 stdout으로 복원합니다.
        sys.stdout = original_stdout

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")