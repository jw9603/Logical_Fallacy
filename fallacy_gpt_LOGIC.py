import random
from openai import OpenAI
import json
from retry import retry
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
from collections import Counter
import math

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
def LOGIC_multi_fallacy_classification_no_query_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ],
    )
    
    return cmpl, mode

@retry()
def LOGIC_multi_fallacy_classification_no_query_zcot_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\nPlease detect a fallacy in the Text.\nLet's think step by step."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ],
    )
    
    return cmpl, mode

@retry()
def LOGIC_multi_fallacy_classification_no_query_zero_def_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\n1. Faulty Generalization : An informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon is an example of jumping to conclusions.\n2. Ad Hominem : An irrelevant attack towards the person or some aspect of the person who is making the argument, instead of addressing the argument or position directly.\n3. False Causality : A statement that jumps to a conclusion implying a causal relationship without supporting evidence.\n4. Ad Populum : A fallacious argument which is based on affirming that something is real or better because the majority thinks so.\n5. Circular Reasoning : A fallacy where the end of an argument comes back to the beginning without having proven itself.\n6. Appeal to Emotion : Manipulation of the recipient’s emotions in order to win an argument.\n7. Deductive Reasoning : An error in the logical structure of an argument.\n8. Red Herring : Also known as red herring, this fallacy occurs when the speaker attempts to divert attention from the primary argument by offering a point that does not suffice as counterpoint/supporting evidence (even if it is true).\n9. Intentional Fallacy : Some intentional/subconscious action/choice to incorrectly support an argument.\n10. False Dilemma : A claim presenting only two options or sides when there are many options or sides.\n11. Irrelevant Authority : An appeal is made to some form of ethics, authority, or credibility.\n12. Fallacy of Extension : An argument that attacks an exaggerated/caricatured version of an opponent’s.\n13. Equivocation : An argument which uses a phrase in an ambiguous way, with one meaning in one portion of the argument and then another meaning in another portion.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ],
    )
    
    return cmpl, mode


@retry()
def LOGIC_multi_fallacy_classification_query_zero_list(text,query):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'.\nPlease detect a fallacy in the Text based on the Query."},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role": "user", "content": "Label: "}
        ],
        logprobs=True
    )
    

    
    return cmpl, mode





@retry()
def LOGIC_multi_fallacy_classification_query_ranking_zero_list(text,response,query1,query2,query3):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-4',
        # model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\nThe ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\nThe label can be 'Faulty Generalization' and 'Ad Hominem' and 'False Causality' and 'Ad populum' and 'Circular Reasoning' and 'Appeal to Emotion' and 'Deductive Reasoning' and 'Red herring' and 'Intentional Fallacy' and 'False Dilemma' and 'Irrelevant Authority' and 'Fallacy of Extension' and 'Equivocation'. Based on the ranking prompt of these queries, please reference them to detect the fallacy in the sentence."},
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
    

    
    with open('./new_data/LOGIC/LOGIC_test.json') as f:
        json_data = json.load(f)
        
    # ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']
    TOTAL_CALLS = len(json_data['test'])

    with open('./result/LOGIC/gpt-3.5-turbo_no_query_result_seed0_1time_5class.txt','w') as output_file:
        import sys
        # 기존의 stdout을 백업합니다.
        original_stdout = sys.stdout
        # 출력을 파일로 리디렉션합니다.
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
           
            
            completion,mode = LOGIC_multi_fallacy_classification_no_query_zero_list(t1)
            # completion,mode = LOGIC_multi_fallacy_classification_no_query_zero_def_list(t1)
            # completion,mode = LOGIC_multi_fallacy_classification_no_query_zcot_list(t1)
            # completion, mode = LOGIC_multi_fallacy_classification_no_query_rethink_zero_list(t1,t6)
            # completion,mode = LOGIC_multi_fallacy_classification_query_zero_list(t1,t5)
            # completion,mode = LOGIC_multi_fallacy_classification_query_ranking_zero_list(t1,t6,t3,t4,t5)
            # completion,mode = LOGIC_multi_fallacy_classification_query_ranking_evidence_zero_list(t1,t6,t7,t8,t9,t3,t4,t5)
            # print('completion',completion)
            pred = completion.choices[0].message.content
            ################## Calculate Confidence Score(3.5) ##########################################
            # completion,mode = LOGIC_multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = LOGIC_multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = LOGIC_multi_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[8] = confidence_score_cg # 8
            # sample[9] = confidence_score_ex # 9
            # sample[10] = confidence_score_go # 10
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # print('final ranking',ranked_prompts)
            # # assert -1 ==0
            # sample[11] = ranked_prompts # 11
            ################### Calculate Confidence Score(3.5) ##########################################
            # ################### Calculate Confidence Score(4) ##########################################
            # completion,mode = LOGIC_multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = LOGIC_multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = LOGIC_multi_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[12] = confidence_score_cg # 12
            # sample[13] = confidence_score_ex # 13
            # sample[14] = confidence_score_go # 14
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # sample[15] = ranked_prompts # 15
            ################### Calculate Confidence Score(4) ##########################################
            
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
        
    # with open('./new_data/LOGIC/LOGIC_test.json','w') as f:

        
    #     json.dump(json_data, f, indent=4)    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")