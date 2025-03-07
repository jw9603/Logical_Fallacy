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


def rank_confidence_scores(confidence_score_cg, confidence_score_ex, confidence_score_go, ranking_prompt):
    # Create a list of tuples containing each confidence score and its associated query type
    score_pairs = [
        (confidence_score_cg, "Counterargument Query"),
        (confidence_score_ex, "Explanation Query"),
        (confidence_score_go, "Goal Query")
    ]

    # Sort the list of tuples in descending order based on confidence scores
    sorted_score_pairs = sorted(score_pairs, key=lambda x: x[0], reverse=True)

    # Extract the query types in the sorted order
    ranked_prompts = [pair[1] for pair in sorted_score_pairs]

    return ranked_prompts
    
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

def generate_response1(system_message, user_message):
    full_prompt = f"{system_message}\n{user_message}"

    inputs = tokenizer(full_prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Generate response
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=True
    )

    # Get the generated tokens and scores
    response_ids = outputs.sequences[0][input_ids.shape[-1]:]
    scores = outputs.scores

    # Calculate log probabilities for each token
    token_logprobs = []
    for i, score in enumerate(scores):
        # Get the probability distribution over the vocabulary
        probs = torch.softmax(score, dim=-1)
        # Get the log probabilities
        log_probs = torch.log(probs)
        # Get the log probability of the generated token
        token_id = response_ids[i]
        token_logprob = log_probs[0, token_id].item()
        token_logprobs.append(token_logprob)

    # Sum the log probabilities to get the total log probability of the sequence
    total_logprob = sum(token_logprobs)

    # Decode the response
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return response_text, response_ids, token_logprobs, total_logprob

def get_label_log_prob(pred, log_probs, labels):
    min_position = float('inf')
    selected_label = 0
    selected_log_prob = 0.0

    for label, code in labels:
        position = pred.find(label)
        if 0 <= position < min_position:
            min_position = position
            selected_label = code
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            start_idx = pred[:position].count(' ')  # calculate start index based on spaces
            end_idx = start_idx + len(label_tokens)
            selected_log_prob = sum(log_probs[start_idx:end_idx])

    return selected_label, selected_log_prob


if __name__ == '__main__':
    CALLS = 0
    
    # Evaluation vectors
    ground_truth = []
    gpt_preds = []
    
    with open('./new_data/Argotario/argotario_test_llama.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']
    # with open('./result/Argotario/llama2/llama2_ex_query_7B.txt','w') as output_file:
    # with open('./result/Argotario/llama2/llama2_prompt_ranking2_7B.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
        
        system_message = (
            "Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion','Faulty Generalization', 'Red Herring', 'Ad Hominem', and 'Irrelevant Authority'.\n"
            "Please detect a fallacy in the Text.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the five labels stated."
        )
        # system_message_zcot = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion','Faulty Generalization', 'Red Herring', 'Ad Hominem', and 'Irrelevant Authority'.\n"
        #     "Please detect a fallacy in the Text.\nLet's think step by step.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the five labels stated."
        # )
        # system_message_def = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion' and 'Faulty Generalization' and 'Red Herring' and 'Ad Hominem' and 'Irrelevant Authority'.\n"
        #     "1. Appeal to Emotion : This fallacy tries to arouse non-rational sentiments within the intended audience in order to persuade.\n"
        #     "2. Faulty Generalization : The argument uses a sample which is too small, or follows falsely from a sub-part to a composite or the other way round.\n"
        #     "3. Red Herring : This argument distracts attention to irrelevant issues away from the thesis which is supposed to be discussed.\n"
        #     "4. Ad Hominem : The opponent attacks a person instead of arguing against the claims that the person has put forward.\n"
        #     "5. Irrelevant Authority : While the use of authorities in argumentative discourse is not fallacious inherently, appealing to authority can be fallacious if the authority is irrelevant to the discussed subject.\n"
        #     "Please detect a fallacy in the Text."
        # )
        # system_message_query = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Appeal to Emotion','Faulty Generalization', 'Red Herring', 'Ad Hominem', and 'Irrelevant Authority'.\n"
        #     "Please detect a fallacy in the Text based on the Query.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the five labels stated."
        # )
        # system_message_pr = (
        #     "Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\n"
        #     "The ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\n"
        #     "The label can be 'Appeal to Emotion','Faulty Generalization', 'Red Herring', 'Ad Hominem', and 'Irrelevant Authority'.\n"
        #     "Based on the ranking prompt of these queries, please reference them to detect the fallacy in the sentence."
        # )
        
        for sample in json_data['test']:
            
            # These are the parameters to be used for running each of the three queries individually, or for Zcot and DEF.
            t1 = 'Text:' +sample[0]
            # t3 = 'Query: '+sample[6] # CG
            # t4 = 'Query: '+sample[7] # EX
            # t5 = 'Query: '+sample[8] # GO
            
        
            # These are the parameters to be used for running each of the prompt ranking
            # t1 = 'Sentence:'+sample[0]
            
            # score_cg = sample[9]
            # score_ex = sample[10]
            # score_go = sample[11]
            # t2 = rank_confidence_scores(score_cg,score_ex,score_go,ranking_prompt)
            # t3 = 'Counterargument Query: '+sample[6] # CG
            # t4 = 'Explanation Query: '+sample[7] # EX
            # t5 = 'Goal Query: '+sample[8] # GO
            
            # t2 = 'Ranking Prompt:' +', '.join(t2)
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
            
            user_message = f"{t1}\nLabel: "
            # user_message_query = f"{t1}\n{t5}\nLabel: "
            # user_message_def =f"{t1}\nLabel: "
            # user_message_zcot=f"{t1}\nPlease analyze step by step to detect any fallacy.\nLabel: "
            # user_message_pr = f"{t1}\n{t2}\n{t3}\n{t4}\n{t5}\nLabel: "
            pred = generate_response(system_message=system_message,user_message=user_message)
            # pred = generate_response(system_message=system_message_def,user_message=user_message_def)
            # pred = generate_response(system_message=system_message_zcot,user_message=user_message_zcot)
            # pred = generate_response(system_message=system_message_query,user_message=user_message_query)
            # pred,response_ids,token_logprobs,total_logprob = generate_response1(system_message=system_message_query,user_message=user_message_query)
            # pred = generate_response(system_message=system_message_pr,user_message=user_message_pr)
            # Determine the label based on the first occurrence of keywords
            # Determine the label based on the first occurrence of keywords
            pred_lower = pred.lower()
            labels = [
                ('appeal to emotion', 1),
                ('faulty generalization', 2),
                ('red herring', 3),
                ('ad hominem', 4),
                ('irrelevant authority', 5)
            ]

            min_position = float('inf')
            selected_label = 0

            for label, code in labels:
                position = pred_lower.find(label)
                if 0 <= position < min_position:
                    min_position = position
                    selected_label = code
            # selected_label, selected_log_prob = get_label_log_prob(pred_lower, token_logprobs, labels)
            gpt_preds.append(selected_label)

            CALLS += 1
            print(CALLS,'/',TOTAL_CALLS)
            print('pred',pred)
            # print('selected_log_prob',selected_log_prob)
            # print('changed',math.exp(selected_log_prob))
            
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
        
        
        # Append the calculated accuracy to the list
        accuracies = []
        for label in [1, 2,3,4,5]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # Print accuracy for Each Class
        print("Class 1 (Appeal to Emotion) Accuracy:", accuracies[0])
        print("Class 2 (Faulty Generalization) Accuracy:", accuracies[1])
        print("Class 3 (Red Herring) Accuracy:", accuracies[2])
        print("Class 4 (Ad Hominem) Accuracy:", accuracies[3])
        print("Class 5 (Irrelevant Authority) Accuracy:", accuracies[4])
      
        
        # Compute Precision, Recall, and F1-Score for each class
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)

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
    
        # Restore the redirected output back to the original stdout.
        sys.stdout = original_stdout
        
    # with open('./new_data/Argotario/argotario_test_sim1_llama.json','w') as f:
    #     json.dump(json_data, f, indent=4)  
    
    print("All output has been saved to 'output.txt' file.")
