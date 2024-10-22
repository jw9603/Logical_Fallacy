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
def multi_fallacy_classification_no_query_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
        ,logprobs=True
    )
    

    
    return cmpl, mode

@retry()
def multi_fallacy_classification_no_query_zcot_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\nPlease detect a fallacy in the Text.\nLet's think step by step."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def multi_fallacy_classification_no_query_zero_def_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n1. Loaded Language : Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.\n2. Exaggeration or Minimisation : Either representing something in an excessive manner: making things larger, worse or making something seem less important than it really is\n3. Doubt : Questioning the credibility of someone or something.\n4. Strawman : When an opponent’s proposition is substituted with a similar one which is then refuted in place of the original proposition.\n5. Flag Waving : Playing on strong national feeling (or to any group) to justify/promote an action/idea.\n6. Thought-Terminating Cliches : Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or distract attention away from other lines of thought.\n7. Appeal to Fear: Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases the support is based on preconceived judgements.\n8. Name Calling: Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises.\n9. Whatboutism: A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument.\n10. False Causality: Assuming a single cause or reason when there are actually multiple causes for an issue.\n11. Irrelevant Authority: Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, although it is referred to as Testimonial in literature.\n12. Slogans: A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.\n13. Reductio Ad Hitlerum:Persuading an audience to disapprove an action or idea by suggesting that the idea is popular with groups hated in contempt by the target audience. It can refer to any person or concept with a negative connotation.\n14. Red Herring: Introducing irrelevant material to the issue being discussed, so that everyone’s attention is diverted away from the points made.\n15. Black and White Fallacy: Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode



@retry()
def multi_fallacy_classification_query_zero_list(text,query):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-4',
        # model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\nPlease detect a fallacy in the Text based on the Query."},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            {"role": "user", "content": "Label: "}
        ]
        ,logprobs=True
    )
    

    
    return cmpl, mode



@retry()
def multi_fallacy_classification_query_ranking_zero_list(text,response,query1,query2,query3):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-4',
        # model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\nThe ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\nThe label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\nBased on the ranking prompt of these queries, please reference them to detect the fallacy in the sentence."},
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


    with open('./new_data/propaganda/propaganda_test.json') as f:
        json_data = json.load(f)
     
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/propaganda/gpt-3.5-turbo_no_query_result_seed0_1time_15class.txt','w') as output_file:

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
            if 'loaded language' == sample[1]:
                ground_truth.append(1)
            elif 'exaggeration or minimisation' == sample[1]:
                ground_truth.append(2)
            elif 'doubt' == sample[1]:
                ground_truth.append(3)
            elif 'strawman' == sample[1]:
                ground_truth.append(4)
            elif 'flag waving' == sample[1]:
                ground_truth.append(5)
            elif 'thought terminating cliches' == sample[1]:
                ground_truth.append(6)
            elif 'appeal to fear' == sample[1]:
                ground_truth.append(7)
            elif 'name calling' == sample[1]:
                ground_truth.append(8)
            elif 'whatboutism' == sample[1]:
                ground_truth.append(9)
            elif 'false causality' == sample[1]:
                ground_truth.append(10)
            elif 'irrelevant authority' == sample[1]:
                ground_truth.append(11)
            elif 'slogans' == sample[1]:
                ground_truth.append(12)
            elif 'reductio ad hitlerum' == sample[1]:
                ground_truth.append(13)
            elif 'red herring' == sample[1]:
                ground_truth.append(14)
            elif 'black and white fallacy' == sample[1]:
                ground_truth.append(15)
    
                
            # Create a completion and a prediction with CHAT-CPT
            # completion, mode = multi_fallacy_classification_query_ranking_zero_list(t1,t6,t3,t4,t5)
            # completion, mode = multi_fallacy_classification_no_query_zcot_list(t1)
            # completion, mode = multi_fallacy_classification_no_query_zero_def_list(t1)
            completion, mode = multi_fallacy_classification_no_query_zero_list(t1)
            # completion, mode = multi_fallacy_classification_query_zero_list(t1,t5)
            pred = completion.choices[0].message.content
            ################### Calculate Confidence Score(3.5) ##########################################
            # completion,mode = multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = multi_fallacy_classification_query_zero_list(t1,t5)
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
            # completion,mode = multi_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = multi_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = multi_fallacy_classification_query_zero_list(t1,t5)
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
           
            
      
            if 'loaded' in pred.lower():
                gpt_preds.append(1)
            elif 'exaggeration' in pred.lower() or 'minimisation' in pred.lower():
                gpt_preds.append(2)
            elif 'doubt' in pred.lower():
                gpt_preds.append(3)
            elif 'strawman' in pred.lower():
                gpt_preds.append(4)
            elif 'flag' in pred.lower():
                gpt_preds.append(5)
            elif 'thought' in pred.lower() or 'cliches' in pred.lower():
                gpt_preds.append(6)
            elif 'fear' in pred.lower() or 'prejudice' in pred.lower():
                gpt_preds.append(7)
            elif 'name' in pred.lower():
                gpt_preds.append(8)
            elif 'whataboutism' in pred.lower():
                gpt_preds.append(9)
            elif 'cause' in pred.lower() or 'causality' in pred.lower():
                gpt_preds.append(10)
            elif 'authority' in pred.lower():
                gpt_preds.append(11)
            elif 'slogans' in pred.lower():
                gpt_preds.append(12)
            elif 'hitlerum' in pred.lower() or 'reductio' in pred.lower():
                gpt_preds.append(13)
            elif 'red herring' in pred.lower():
                gpt_preds.append(14)
            elif 'black and white' in pred.lower():
                gpt_preds.append(15)
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
        for label in [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # 클래스별 정확도 출력
        print("Class 1 (Loaded Language) Accuracy:", accuracies[0])
        print("Class 2 (Exaggeration or Minimisation) Accuracy:", accuracies[1])
        print("Class 3 (Doubt) Accuracy:", accuracies[2])
        print("Class 4 (Strawman) Accuracy:", accuracies[3])
        print("Class 5 (Flag Waving) Accuracy:", accuracies[4])
        print("Class 6 (Thought terminating Cliches) Accuracy:", accuracies[5])
        print("Class 7 (Appeal to Fear) Accuracy:", accuracies[6])
        print("Class 8 (Name Calling) Accuracy:", accuracies[7])
        print("Class 9 (Whatboutism) Accuracy:", accuracies[8])
        print("Class 10 (False Causality) Accuracy:", accuracies[9])
        print("Class 11 (Irrelevant Authority) Accuracy:", accuracies[10])
        print("Class 12 (Slogans) Accuracy:", accuracies[11])
        print("Class 13 (Reductio Ad Hitlerum) Accuracy:", accuracies[12])
        print("Class 14 (Red Herring) Accuracy:", accuracies[13])
        print("Class 15 (Black and White Fallacy) Accuracy:", accuracies[14])
      
        
        # 클래스별로 Precision, Recall, F1-Score 계산
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)
        # 클래스별 결과 출력
        classes = ["Loaded Language","Exaggeration or Minimisation","Douby","Strawman","Flag Waving","Thought terminating Cliches","Appeal to Fear","Name Calling","Whatboutism","False Causality","Irrelevant Authority","Slogans","Reductio Ad Hitlerum","Red Herring","Black and White Fallacy"]
        classes1 = ["The Other","Loaded Language","Exaggeration or Minimisation","Douby","Strawman","Flag Waving","Thought terminating Cliches","Appeal to Fear","Name Calling","Whatboutism","False Causality","Irrelevant Authority","Slogans","Reductio Ad Hitlerum","Red Herring","Black and White Fallacy"]
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
        
        
    # with open('./new_data/propaganda/propaganda_test.json','w') as f: # Uncomment this line to save the confidence score.

        
        # json.dump(json_data, f, indent=4)    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")
