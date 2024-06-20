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
    
    with open('./new_data/propaganda/propaganda_test_with_no_fallacy.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./result/propaganda/llama2/binary/llama2_promptranking_7B.txt','w') as output_file:
 
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
        # system_message_pr = (
        #     "Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\n"
        #     "The ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\n"
        #     "The label can be 'Fallacy' or 'None'.\n"
        #     "Based on the ranking prompt of these queries, please reference them to detect if a fallacy exists in the sentence."
        # )
        
        for sample in json_data['test']:
            # These are the parameters to be used for running each of the three queries individually, or for Zcot and DEF.
            t1 = 'Text:' +sample[0]
            # t3 = 'Query: '+sample[5] # CG
            # t4 = 'Query: '+sample[6] # EX
            # t5 = 'Query: '+sample[7] # GO
            # label = sample[1]
            
            
            #  These are the parameters to be used for running each of the prompt ranking
            # t1 = 'Sentence:'+sample[0]
            
            # score_cg = sample[9]
            # score_ex = sample[10]
            # score_go = sample[11]
            # t2 = rank_confidence_scores(score_cg,score_ex,score_go,ranking_prompt)
            # t2 = 'Ranking Prompt:' +', '.join(sample[15])
            # t3 = 'Counterargument Query: '+sample[5] # CG
            # t4 = 'Explanation Query: '+sample[6] # EX
            # t5 = 'Goal Query: '+sample[7] # GO
            
            # t2 = 'Ranking Prompt:' +', '.join(t2)
    
            # Label
            if 'loaded language' == sample[1]:
                defin = "Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience."
                ground_truth.append(1)
            elif 'exaggeration or minimisation' == sample[1]:
                defin = "Either representing something in an excessive manner: making things larger, worse or making something seem less important than it really is."
                ground_truth.append(1)
            elif 'doubt' == sample[1]:
                defin = "Questioning the credibility of someone or something."
                ground_truth.append(1)
            elif 'strawman' == sample[1]:
                defin = "When an opponent’s proposition is substituted with a similar one which is then refuted in place of the original proposition."
                ground_truth.append(1)
            elif 'flag waving' == sample[1]:
                defin = "Playing on strong national feeling (or to any group) to justify/promote an action/idea."
                ground_truth.append(1)
            elif 'thought terminating cliches' == sample[1]:
                defin = "Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or distract attention away from other lines of thought."
                ground_truth.append(6)
            elif 'appeal to fear' == sample[1]:
                defin = "Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases the support is based on preconceived judgements."
                ground_truth.append(1)
            elif 'name calling' == sample[1]:
                defin = "Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises."
                ground_truth.append(1)
            elif 'whatboutism' == sample[1]:
                defin = "A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument."
                ground_truth.append(1)
            elif 'false causality' == sample[1]:
                defin = "Assuming a single cause or reason when there are actually multiple causes for an issue."
                ground_truth.append(1)
            elif 'irrelevant authority' == sample[1]:
                defin = "Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, although it is referred to as Testimonial in literature."
                ground_truth.append(1)
            elif 'slogans' == sample[1]:
                defin = "A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals."
                ground_truth.append(1)
            elif 'reductio ad hitlerum' == sample[1]:
                defin = "Persuading an audience to disapprove an action or idea by suggesting that the idea is popular with groups hated in contempt by the target audience. It can refer to any person or concept with a negative connotation."
                ground_truth.append(1)
            elif 'red herring' == sample[1]:
                defin = "Introducing irrelevant material to the issue being discussed, so that everyone’s attention is diverted away from the points made."
                ground_truth.append(1)
            elif 'black and white fallacy' == sample[1]:
                defin = "Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship)."
                ground_truth.append(1)
            elif 'no fallacy' == sample[1]:
                defin = "non propaganda"
                ground_truth.append(2)
            # system_message_def = (
            # f"Your task is to detect the fallacy known as {label} in the Text. This is the definition for {label}: {defin}.\n"
            # "The text's label can be 'Fallacy' or 'None'.\nPlease ensure your response is either 'Fallacy' or 'None'."
            # )
            
 
            user_message = f"{t1}\nLabel: "
            # user_message_def = f"{t1}\nPlease analyze the text to detect the fallacy known as {sample[1]}, which is defined as: {defin}.\nLabel: "
            # user_message_zcot = f"{t1}\nPlease analyze step by step to detect any fallacy.\nLabel: "
            # user_message_query = f"{t1}\n{t5}\nLabel: "
            # user_message_pr = f"{t1}\n{t2}\n{t3}\n{t4}\n{t5}\nLabel: "
            # pred = generate_response(system_message=system_message,user_message=user_message)
            # pred = generate_response(system_message=system_message_def,user_message=user_message_def)
            # pred = generate_response(system_message=system_message_zcot,user_message=user_message_zcot)
            # pred = generate_response(system_message=system_message_query,user_message=user_message_query)
            pred = generate_response(system_message=system_message_pr,user_message=user_message_pr)
            # Determine the label based on the first occurrence of keywords
            pred_lower = pred.lower()
            labels = [
                ('loaded language', 1),
                ('exaggeration or minimisation', 1),
                ('doubt', 1),
                ('strawman', 1),
                ('flag waving', 1),
                ('thought terminating cliches', 1),
                ('appeal to fear', 1),
                ('name calling', 1),
                ('whataboutism', 1),
                ('false causality', 1),
                ('irrelevant authority', 1),
                ('slogans', 1),
                ('reductio ad hitlerum', 1),
                ('red herring', 1),
                ('black and white fallacy', 1),
                ('fallacy', 1),  # Fallacy detection code
                ('no', 2),
                ('not', 2),
                ('none', 2)
            ]

            min_position = float('inf')
            selected_label = 0

            for label, code in labels:
                position = pred_lower.find(label)
                if 0 <= position < min_position:
                    min_position = position
                    selected_label = code

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