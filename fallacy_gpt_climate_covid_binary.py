import random
from openai import OpenAI
import json
from retry import retry
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
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
def binary_fallacy_classification_no_query_zero_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        model='gpt-4',
        # model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode
@retry()
def binary_fallacy_classification_no_query_zcot_list(text):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-4',
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\nPlease detect a fallacy in the Text.\nLet's think step by step."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def binary_fallacy_classification_no_query_def_list(text,defin,label):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4',
        messages=[
            {"role":"system",
             "content":"Your are a trained model capable of identifying the logical fallacy known as {label}.\nThis is the definition for {label}:{defin}.\nThe text's label can be 'Fallacy' or 'None'.\nPlease detect a fallacy in the Text."},
            {"role": "user", "content": text},
            {"role": "user", "content": "Label: "}
        ]
    )
    

    
    return cmpl, mode

@retry()
def binary_fallacy_classification_query_zero_list(text,query):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-4',
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\nPlease detect a fallacy in the Text based on the Query.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the two labels stated."},
            {"role": "user", "content": text},
            {"role": "user", "content": query},
            # {"role": "user", "content": text2},
            # {"role": "user", "content": text3},
            {"role": "user", "content": "Label: "}
        ],logprobs=True
    )
    

    
    return cmpl, mode

@retry()
def binary_fallacy_classification_query_ranking_zero_list(text,response,query1,query2,query3):
    mode = 'ZERO-SHOT'
    cmpl = client.chat.completions.create(
        # model='gpt-4',
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system",
             "content":"Given a sentence, we aim to detect the presence of a logical fallacy using queries based on multiple perspectives, such as counterargument, explanation, and goal.\nThe ranking prompt indicates the order of queries based on their confidence scores, which help determine whether a logical fallacy is present in the sentence.\nThe label can be 'Fallacy' or 'None'. Based on the ranking prompt of these queries, please reference them to detect if a fallacy exists in the sentence."},
            {"role": "user", "content": text},
            {"role": "user", "content": response},
            {"role": "user", "content": query1},
            {"role": "user", "content": query2},
            {"role": "user", "content": query3},
            {"role": "user", "content": "Label: "}

        ]
    )
    return cmpl, mode

    
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
    with open('./new_data/CLIMATE/climate_test_with_no_fallacy.json') as f:
    # with open('./new_data/CLIMATE/climate_test.json') as f:
        json_data = json.load(f)
    # ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']
    TOTAL_CALLS = len(json_data['test'])
    # with open('./result/COVID-19/binary/gpt-3.5-turbo_no_query_result_seed0_1time.txt','w') as output_file:
    with open('./result/CLIMATE/binary/gpt-3.5-turbo_confidence_ranking_result_seed0_1time_.txt','w') as output_file:
        import sys
        # 기존의 stdout을 백업합니다.
        original_stdout = sys.stdout
        # 출력을 파일로 리디렉션합니다.ㅌ
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
            if 'cherry picking' == sample[1]:
                defin = "The act of choosing among competing evidence that which supports a given position, ignoring or dismissing findings which do not support it."
                ground_truth.append(1)
            elif 'vagueness' == sample[1]:
                defin = "A word/a concept or a sentence structure which are ambiguous are shifted in meaning in the process of arguing or are left vague being potentially subject to skewed interpretations."
                ground_truth.append(1)
            elif 'red herring' == sample[1]:
                defin = "The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand."
                ground_truth.append(1)
            elif 'false causality' == sample[1]:
                defin = "1. A generalization is drawn from a sample which is too small, not representative or not applicable to the situation if all the variables are taken into account. 2. X is identified as the cause of Y when another factor Z causes both X and Y OR X is considered the cause of Y when actually it is the opposite "
                ground_truth.append(1)
            elif 'irrelevant authority' == sample[1]:
                defin = "An appeal to authority is made where the it lacks credibility or knowledge in the discussed matter or the authority is attributed a tweaked statement."
                ground_truth.append(1)
            elif 'evading the burden of proof' == sample[1]:
                defin = "A position is advanced without any support as if it was self-evident."
                ground_truth.append(1)
            elif 'strawman' == sample[1]:
                defin = "The arguer misinterprets an opponent’s argument for the purpose of more easily attacking it, demolishes the misinterpreted argument, and then proceeds to conclude that the opponent’s real argument has been demolished."
                ground_truth.append(1)
            elif 'false analogy' == sample[1]:
                defin = "because two things [or situations] are alike in one or more respects, they are necessarily alike in some other respect."
                ground_truth.append(1)
            elif 'faulty generalization' == sample[1]:
                defin = "A generalization is drawn from a sample which is too small, not representative or not applicable to the situation if all the variables are taken into account."
                ground_truth.append(1)
            elif 'no fallacy' == sample[1]:
                defin = "no fallacy"
                ground_truth.append(2)

            ################### Calculate Confidence Score(3.5) ##########################################
            # completion,mode = binary_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = binary_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = binary_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[9] = confidence_score_cg # 9
            # sample[10] = confidence_score_ex # 10
            # sample[11] = confidence_score_go # 11
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # print('final ranking',ranked_prompts)
            # # assert -1 ==0
            # sample[12] = ranked_prompts # 12
            ################### Calculate Confidence Score(3.5) ##########################################
            # ################### Calculate Confidence Score(4) ##########################################
            # completion,mode = binary_fallacy_classification_query_zero_list(t1,t3)
            # pred = completion.choices[0].message.content
            # logprobs = completion.choices[0].logprobs
            # total_logprobs = [logprobs.content[i].logprob for i in range(len(logprobs.content))]
            # confidence_score_cg = sum(total_logprobs)
            # print('pred_cg',pred)
            # print('confidence_score_cg',confidence_score_cg)
            
            # completion1,mode1 = binary_fallacy_classification_query_zero_list(t1,t4)
            # pred1 = completion1.choices[0].message.content
            # logprobs1 = completion1.choices[0].logprobs
            # total_logprobs1 = [logprobs1.content[i].logprob for i in range(len(logprobs1.content))]
            # confidence_score_ex = sum(total_logprobs1)
            # print('pred_ex',pred1)
            # print('confidence_score_ex',confidence_score_ex)
            
            # completion2,mode2 = binary_fallacy_classification_query_zero_list(t1,t5)
            # pred2 = completion2.choices[0].message.content
            # logprobs2 = completion2.choices[0].logprobs
            # total_logprobs2 = [logprobs2.content[i].logprob for i in range(len(logprobs2.content))]
            # confidence_score_go = sum(total_logprobs2)
            # print('pred_go',pred)
            # print('confidence_score_go',confidence_score_go)

            # sample[13] = confidence_score_cg # 13
            # sample[14] = confidence_score_ex # 14
            # sample[15] = confidence_score_go # 15
            # ranked_prompts = rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt)
            # sample[16] = ranked_prompts # 16
            ################### Calculate Confidence Score(4) ##########################################
                
            # Create a completion and a prediction with CHAT-CPT
            # completion, mode = binary_fallacy_classification_query_ranking_zero_list(t1,t6,t3,t4,t5)
            # completion,mode = binary_fallacy_classification_no_query_zcot_list(t1)
            completion,mode = binary_fallacy_classification_no_query_zero_list(t1)
            # completion,mode = binary_fallacy_classification_no_query_def_list(t1,defin,sample[1])
            # completion,mode = binary_fallacy_classification_query_zero_list(t1,t5)
       
            pred = completion.choices[0].message.content
            
            if 'picking' in pred.lower():
                gpt_preds.append(1)
            elif 'vagueness' in pred.lower():
                gpt_preds.append(1)
            elif 'red herring' in pred.lower():
                gpt_preds.append(1)
            elif 'causality' in pred.lower() or 'cause' in pred.lower():
                gpt_preds.append(1)
            elif 'authority' in pred.lower():
                gpt_preds.append(1)
            elif 'evading' in pred.lower():
                gpt_preds.append(1)
            elif 'strawman' in pred.lower():
                gpt_preds.append(1)
            elif 'analogy' in pred.lower():
                gpt_preds.append(1)
            elif 'generalization' in pred.lower():
                gpt_preds.append(1)
            elif 'fallacy' in pred.lower():
                gpt_preds.append(1)
            elif 'no' in pred.lower() or 'not' or 'none' in pred.lower():
                gpt_preds.append(2)
            else:
                gpt_preds.append(1)
                    
        

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
        
    
        # 파일로 리디렉션된 출력을 다시 기존 stdout으로 복원합니다.
        sys.stdout = original_stdout
        
    # with open('./new_data/COVID-19/covid_test_with_no_fallacy.json','w') as f:

        
    #     json.dump(json_data, f, indent=4)  

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")