import random
import json
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
import re
import math
from collections import Counter
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

os.environ['HF_TOKEN'] = 'YOURKEY'

# Initialize the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the correct model path if different
# model_name ="meta-llama/Llama-2-13b-hf"
# model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    device_map='auto',
)


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

def generate_response(system_message, user_message):
    full_prompt = f"{system_message}\n{user_message}"

    inputs = tokenizer(full_prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Add attention mask
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,  # Adjusted top_p for better sampling
        temperature=0.7,  # Adjusted temperature for less deterministic output
    )

    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True).strip()

if __name__ == '__main__':
    CALLS = 0
    
    # Evaluation vectors
    ground_truth = []
    gpt_preds = []
    
    # with open('./new_data/CLIMATE/climate_test_with_no_fallacy.json') as f:
        # json_data = json.load(f)
    with open('./new_data/COVID-19/covid_test_with_no_fallacyjson') as f:
        json_data = json.load(f)
    ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']    
    TOTAL_CALLS = len(json_data['test'])
    # with open('./result/COVID-19/binary/llama/llama3_no_query.txt','w') as output_file:


        
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
        
        system_message = (
            "Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\n"
            "Please detect a fallacy in the Text.\nPlease ensure your response is either 'Fallacy' or 'None'."
        )
        # system_message_zcot = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\n"
        #     "Please detect a fallacy in the Text.\nLet's think step by step.\nPlease ensure your response is either 'Fallacy' or 'None'."
        # )
        # system_message_query = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Fallacy' or 'None'.\n"
        #     "Please detect a fallacy in the Text based on the Query.\nPlease ensure your response is either 'Fallacy' or 'None'."
        # )
   
        
        for sample in json_data['test']:
            t1 = 'Text:' +sample[0]
            # t3 = 'Query: '+sample[6] # CG
            # t4 = 'Query: '+sample[7] # EX
            # t5 = 'Query: '+sample[8] # GO
            # label = sample[1]
           
            
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
            # system_message_def = (
            # f"Your task is to detect the fallacy known as {label} in the Text. This is the definition for {label}: {defin}.\n"
            # "The text's label can be 'Fallacy' or 'None'.\nPlease ensure your response is either 'Fallacy' or 'None'."
            # )
            
            user_message = f"{t1}\nLabel: "
            # user_message_def = f"{t1}\nPlease analyze the text to detect the fallacy known as {sample[1]}, which is defined as: {defin}.\nLabel: "
            # user_message_zcot = f"{t1}\nPlease analyze step by step to detect any fallacy.\nLabel: "
            # user_message_query = f"{t1}\n{t5}\nLabel: "
  
            pred = generate_response(system_message=system_message,user_message=user_message)
            # pred = generate_response(system_message=system_message_def,user_message=user_message_def)
            # pred = generate_response(system_message=system_message_zcot,user_message=user_message_zcot)
            # pred = generate_response(system_message=system_message_query,user_message=user_message_query)
            # pred,response_ids,token_logprobs,total_logprob = generate_response1(system_message=system_message_query,user_message=user_message_query)

            # Convert the prediction to lowercase for case-insensitive matching
            pred_lower = pred.lower()

            # Define a list of labels and their corresponding codes
            labels = [
                ('picking', 1),
                ('vagueness', 1),
                ('red herring', 1),
                ('causality', 1),
                ('cause', 1),  # Merged with causality for the same effect
                ('authority', 1),
                ('evading', 1),
                ('strawman', 1),
                ('analogy', 1),
                ('generalization', 1),
                ('fallacy', 1),  # Fallacy detection code
                ('no', 2),
                ('not', 2),
                ('none', 2)  # Grouped 'no', 'not', 'none' for no fallacy detected
            ]

            # # Initialize the variables to determine the first occurring keyword and its label
            min_position = float('inf')
            selected_label = 0

            # # Iterate through the labels and their codes to find the first occurrence
            for label, code in labels:
                position = pred_lower.find(label)
                if 0 <= position < min_position:
                    min_position = position
                    selected_label = code

            gpt_preds.append(selected_label)

            CALLS += 1
            print(CALLS,'/',TOTAL_CALLS)
            print('pred',pred)
  
            
            # sample[11] = selected_log_prob
            
        # Overall Accuracy
        print('ground_truth',ground_truth)
        print('ground_truth길이',len(ground_truth))
        # print('confidences',confidences)
        # print('confidences길이',len(confidences))
        
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