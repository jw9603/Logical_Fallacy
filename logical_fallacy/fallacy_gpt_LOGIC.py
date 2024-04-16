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
def LOGIC_multi_fallacy_classification_no_query_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode


@retry()
def LOGIC_multi_fallacy_classification_query_zero_list(text,query1,query2):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        # model='gpt-3.5-turbo-16k-0613',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\nPlease detect a fallacy in the Text based on the Queries."},
            {"role": "user", "content": text},
            {"role": "user", "content": query1},
            {"role": "user", "content": query2},
            # {"role": "user", "content": query3},
            # {"role": "user", "content": query4},
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
    
    with open('./new_data/LOGIC/LOGIC_test.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/LOGIC/gpt-3.5-turbo_cg_ex_query_result_seed0_1time_13class.txt','w') as output_file:
        import sys
        # 기존의 stdout을 백업합니다.
        original_stdout = sys.stdout
        # 출력을 파일로 리디렉션합니다.ㅌ
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
            t1 = 'Text: '+sample[0]
            # t2 = 'Query1:' +sample[5] # 관계 기반 Query
            t3 = 'Query1:' +sample[6] # 반론 기반 Query
            t4 = 'Query2:' +sample[7] # 설명 기반 Query
            # t5 = 'Query2:' +sample[8] # 목표 기반 query
            # t6 = 'Reread the Text: '+sample[0]
            
            # Label
            if 'faulty generalization' == sample[1]:
                ground_truth.append(1)
            elif 'ad hominem' == sample[1]:
                ground_truth.append(2)
            elif 'false causality' == sample[1]:
                ground_truth.append(3)
            elif 'ad populum' == sample[1]:
                ground_truth.append(4)
            elif 'circular reasoning' == sample[1]:
                ground_truth.append(5)
            elif 'appeal to emotion' == sample[1]:
                ground_truth.append(6)
            elif 'deductive reasoning' == sample[1]:
                ground_truth.append(7)
            elif 'red herring' == sample[1]:
                ground_truth.append(8)
            elif 'intentional fallacy' == sample[1]:
                ground_truth.append(9)
            elif 'false dilemma' == sample[1]:
                ground_truth.append(10)
            elif 'irrelevant authority' == sample[1]:
                ground_truth.append(11)
            elif 'fallacy of extension' == sample[1]:
                ground_truth.append(12)
            elif 'equivocation' == sample[1]:
                ground_truth.append(13)
            
            # elif 'no fallacy' == sample[1]:
                # ground_truth.append(4)
            # else:
            #     ground_truth.append(0)
                
            # Create a completion and a prediction with CHAT-CPT
           
           
            # completion,mode = LOGIC_multi_fallacy_classification_no_query_zero_list(t1)
            # completion, mode = LOGIC_multi_fallacy_classification_no_query_rethink_zero_list(t1,t6)
            completion,mode = LOGIC_multi_fallacy_classification_query_zero_list(t1,t3,t4)
            pred = completion.choices[0].message.content
            
            if 'generalization' in pred.lower():
                gpt_preds.append(1)
            elif 'hominem' in pred.lower():
                gpt_preds.append(2)
            elif 'causality' in pred.lower() or 'cause' in pred.lower():
                gpt_preds.append(3)
            elif 'populum' in pred.lower():
                gpt_preds.append(4)
            elif 'circle' in pred.lower() or 'circular' in pred.lower():
                gpt_preds.append(5)
            elif 'deductive' in pred.lower():
                gpt_preds.append(6)
            elif 'red herring' in pred.lower():
                gpt_preds.append(7)
            elif 'intention' in pred.lower():
                gpt_preds.append(8)
            elif 'dilemma' in pred.lower():
                gpt_preds.append(9)
            elif 'emotion' in pred.lower():
                gpt_preds.append(10)
            elif 'extension' in pred.lower():
                gpt_preds.append(11)
            elif 'authority' in pred.lower():
                gpt_preds.append(12)
            elif 'equivocation' in pred.lower():
                gpt_preds.append(13)
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
        for label in [1, 2,3,4,5,6,7,8,9,10,11,12,13]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # 클래스별 정확도 출력
        print("Class 1 (Faulty Generalization) Accuracy:", accuracies[0])
        print("Class 2 (Ad Hominem) Accuracy:", accuracies[1])
        print("Class 3 (False Causality) Accuracy:", accuracies[2])
        print("Class 4 (Ad Populum) Accuracy:", accuracies[3])
        print("Class 5 (Circular Reasoning) Accuracy:", accuracies[4])
        print("Class 6 (Appeal to Emotion) Accuracy:", accuracies[5])
        print("Class 7 (Deductive Reasoning) Accuracy:", accuracies[6])
        print("Class 8 (Red Herring) Accuracy:", accuracies[7])
        print("Class 9 (Intentional Fallacy) Accuracy:", accuracies[8])
        print("Class 10 (False Dilemma) Accuracy:", accuracies[9])
        print("Class 11 (Irrelevant Authority) Accuracy:", accuracies[10])
        print("Class 12 (Fallacy of Extension) Accuracy:", accuracies[11])
        print("Class 13 (Equivocation) Accuracy:", accuracies[12])
        
        # 클래스별로 Precision, Recall, F1-Score 계산
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)
        # 클래스별 결과 출력
        classes = ["Faulty Generalization","Ad Hominem","False Causality","Ad Populum","Circular Reasoning","Appeal to Emotion","Deductive Reasoning","Red Herring","Intentional Fallacy","False Dilemma","Irrelevant Authority","Fallacy of Extension","Equivocation"]
        classes1 = ["The Other","Faulty Generalization","Ad Hominem","False Causality","Ad Populum","Circular Reasoning","Appeal to Emotion","Deductive Reasoning","Red Herring","Intentional Fallacy","False Dilemma","Irrelevant Authority","Fallacy of Extension","Equivocation"]
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