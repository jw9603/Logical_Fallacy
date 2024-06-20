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
# model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the correct model path if different
# model_name ="meta-llama/Llama-2-13b-hf"
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    device_map='auto',
)

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
    
    with open('./new_data/Argotario/argotario_test_with_no_fallacy.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/Argotario/binary/llama/llama2_no_def_7B.txt','w') as output_file:
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
            label = sample[1]
            
            # Label
            if 'appeal to emotion' == sample[1]:
                defin = "This fallacy tries to arouse non-rational sentiments within the intended audience in order to persuade."
                ground_truth.append(1)
            elif 'faulty generalization' == sample[1]:
                defin = "The argument uses a sample which is too small, or follows falsely from a sub-part to a composite or the other way round."
                ground_truth.append(1)
            elif 'red herring' == sample[1]:
                defin = "This argument distracts attention to irrelevant issues away from the thesis which is supposed to be discussed."
                ground_truth.append(1)
            elif 'ad hominem' == sample[1]:
                defin = "The opponent attacks a person instead of arguing against the claims that the person has put forward."
                ground_truth.append(1)
            elif 'irrelevant authority' == sample[1]:
                defin = "While the use of authorities in argumentative discourse is not fallacious inherently, appealing to authority can be fallacious if the authority is irrelevant to the discussed subject."
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
            
            # Convert the prediction to lowercase for case-insensitive matching
            pred_lower = pred.lower()

            # Define a list of labels and their corresponding codes
            labels = [
                ('emotion', 1),  # Code 1 for 'emotion'
                ('generalization', 1),  # Same code 1 for 'generalization'
                ('red herring', 1),  # Same code 1 for 'red herring'
                ('hominem', 1),  # Same code 1 for 'ad hominem'
                ('authority', 1),  # Same code 1 for 'irrelevant authority'
                ('no', 2),  # Code 2 for 'no'
                ('not', 2),  # Same code 2 for 'not'
                ('none', 2)  # Same code 2 for 'none'
            ]

            # Initialize the variables to determine the first occurring keyword and its label
            min_position = float('inf')
            selected_label = 0

            # Iterate through the labels and their codes to find the first occurrence
            for label, code in labels:
                position = pred_lower.find(label)
                if 0 <= position < min_position:
                    min_position = position
                    selected_label = code

            # Append the found label to the predictions list
            gpt_preds.append(selected_label)

            CALLS += 1
            print(CALLS,'/',TOTAL_CALLS)
            print('pred',pred)
            
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
     

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")
