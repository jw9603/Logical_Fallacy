import random
from openai import OpenAI
import json
from retry import retry
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
import re
import math
from collections import Counter



def rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt):
    # 각 신뢰도 점수와 연관된 쿼리 유형을 튜플 리스트로 생성
    score_pairs = [
        (confidence_score_cg, "Counterargument Query"),
        (confidence_score_ex, "Explanation Query"),
        (confidence_score_go, "Goal Query")
    ]

    # 신뢰도 점수에 따라 내림차순으로 튜플 리스트를 정렬
    sorted_score_pairs = sorted(score_pairs, key=lambda x: x[0], reverse=True)

    # 정렬된 순서대로 쿼리 유형을 추출
    ranked_prompts = [pair[1] for pair in sorted_score_pairs]

    return ranked_prompts

@retry()
def Argotario_multi_fallacy_classification_no_query_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
        ,logprobs=True
    )
    

    
    return cmpl, mode

@retry()
def Argotario_multi_fallacy_classification_no_query_zcot_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text.\nLet's think step by step."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def Argotario_multi_fallacy_classification_no_query_zero_def_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'.\n1. Appeal to Emotion : This fallacy tries to arouse non-rational sentiments within the intended audience in order to persuade.\n2. Faulty Generalization : The argument uses a sample which is too small, or follows falsely from a sub-part to a composite or the other way round.\n3. Red Herring : This argument distracts attention to irrelevant issues away from the thesis which is supposed to be discussed.\n4. Ad Hominem : The opponent attacks a person instead of arguing against the claims that the person has put forward.\n5. Irrelevant Authority : While the use of authorities in argumentative discourse is not fallacious inherently, appealing to authority can be fallacious if the authority is irrelevant to the discussed subject.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode



@retry()
def Argotario_multi_fallacy_classification_query_zero_list(text,query):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-4',
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'.\nPlease detect a fallacy in the Text based on the Query."},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role": "user", "content": "Label: "}
        ]
        ,logprobs=True
    )
    

    
    return cmpl, mode






@retry()
def Argotario_multi_fallacy_classification_query_ranking_zero_list(text,response,query1,query2,query3):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-4',
        # model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\nThe ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\nThe label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'. Based on the ranking prompt of these queries, please reference them to detect the fallacy in the sentence."},
            {"role": "user", "content": text},
            {"role": "user", "content": response},
            {"role": "user", "content": query1},
            {"role": "user", "content": query2},
            {"role": "user", "content": query3},
            {"role": "user", "content": "Label: "}

        ]
    )
    return cmpl, mode




if __name__ =='__main__':

    CALLS = 0  # Initialize the API call counter
    mode = 'None'
    
    client = OpenAI(
    api_key="YOURKEY"
    )
    
    # Evaluation vectors
    ground_truth = []
    gpt_preds = []
    confidences = []
    
    
    with open('./new_data/Argotario/argotario_test.json') as f:
        json_data = json.load(f)
     
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/Argotario/gpt-3.5-turbo_no_query_result_seed0_1time_5class.txt','w') as output_file:
        import sys

        original_stdout = sys.stdout

        sys.stdout = output_file
        for sample in json_data['test']:
            # These are the parameters to be used for running each of the prompt ranking
            # t1 = 'Sentence: '+sample[0]
            # t3 = 'Counterargument Query:' +sample[5] # CG Query
            # t4 = 'Explanation Query:' +sample[6] # EX Query
            # t5 = 'Goal Query:' +sample[7] # GO query
            # t6 = 'Ranking Prompt:' +', '.join(sample[11]) # It is gpt-3.5-turbo's prompt ranking
            # t6 = 'Ranking Prompt:' +', '.join(sample[15]) # It is gpt-4's prompt ranking

    
        
            # These are the parameters to be used for running each of the three queries individually, or for Zcot and DEF.
            t1 = 'Text: '+sample[0]
            # t3 = 'Query: '+sample[5] # CG
            # t4 = 'Query: '+sample[6] # EX
            # t5 = 'Query: '+sample[7] # GO
        
            
            # Label
            if 'appeal to emotion' == sample[1]:
                ground_truth.append(1)
            elif 'faulty generalization' == sample[1]:
                ground_truth.append(2)
            elif 'red herring' == sample[1]:
                ground_truth.append(3)
            elif 'ad hominem' == sample[1]:
                ground_truth.append(4)
            elif 'irrelevant authority' == sample[1]:
                ground_truth.append(5)
    
                
            # Create a completion and a prediction with CHAT-CPT
           
            completion, mode = Argotario_multi_fallacy_classification_no_query_zero_list(t1)
            # completion, mode = Argotario_multi_fallacy_classification_no_query_zcot_list(t1)
            # completion, mode = Argotario_multi_fallacy_classification_no_query_zero_def_list(t1)
            # completion, mode = Argotario_multi_fallacy_classification_query_ranking_zero_list(t1,t6,t3,t4,t5)
            # completion, mode = Argotario_multi_fallacy_classification_query_ranking_evidence_zero_list(t1,t6,t7,t8,t9,t3,t4,t5)
            # completion,mode = Argotario_multi_fallacy_classification_query_zero_list(t1,t5)
            pred = completion.choices[0].message.content
            ################### Calculate Confidence Score(3.5) ##########################################
            # completion,mode = Argotario_multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = Argotario_multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = Argotario_multi_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[20] = confidence_score_cg # 20
            # sample[21] = confidence_score_ex # 21
            # sample[22] = confidence_score_go # 22
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # print('final ranking',ranked_prompts)
            # # assert -1 ==0
            # sample[23] = ranked_prompts # 23
            ################### Calculate Confidence Score(3.5) ##########################################
            # ################### Calculate Confidence Score(4) ##########################################
            # completion,mode = Argotario_multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = Argotario_multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = Argotario_multi_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[24] = confidence_score_cg # 24
            # sample[25] = confidence_score_ex # 25
            # sample[26] = confidence_score_go # 26
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # sample[27] = ranked_prompts # 27
            ################### Calculate Confidence Score(4) ##########################################
           
            
      
            if 'emotion' in pred.lower():
                gpt_preds.append(1)
            elif 'generalization' in pred.lower():
                gpt_preds.append(2)
            elif 'red herring' in pred.lower():
                gpt_preds.append(3)
            elif 'hominem' in pred.lower():
                gpt_preds.append(4)
            elif 'authority' in pred.lower():
                gpt_preds.append(5)
            else:
                gpt_preds.append(0)
                    
        

            CALLS += 1
 
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
        for label in [1, 2,3,4,5]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # 클래스별 정확도 출력
        print("Class 1 (Appeal to Emotion) Accuracy:", accuracies[0])
        print("Class 2 (Faulty Generalization) Accuracy:", accuracies[1])
        print("Class 3 (Red Herring) Accuracy:", accuracies[2])
        print("Class 4 (Ad Hominem) Accuracy:", accuracies[3])
        print("Class 5 (Irrelevant Authority) Accuracy:", accuracies[4])
      
        
        # 클래스별로 Precision, Recall, F1-Score 계산
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)
        # 클래스별 결과 출력
        classes = ["Appeal to Emotion","Faulty Generalization","Red Herring","Ad Hominem","Irrelevant Authority"]
        classes1 = ["The Other","Appeal to Emotion","Faulty Generalization","Red Herring","Ad Hominem","Irrelevant Authority"]
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
        
        
    # with open('./new_data/Argotario/argotario_test_sim1.json','w') as f:

        
    #     json.dump(json_data, f, indent=4)    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")
