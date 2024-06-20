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
model_name ="meta-llama/Llama-2-13b-hf"
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
    
    # with open('./new_data/CLIMATE/climate_test.json') as f:
        # json_data = json.load(f)
    with open('./new_data/COVID-19/covid_test.json') as f:
        json_data = json.load(f)
        
    ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']
    TOTAL_CALLS = len(json_data['test'])
    
    with open('./result/COVID-19/llama2/llama2_prompt_ranking8_13B.txt','w') as output_file:
        
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
        
        system_message = (
            "Your task is to detect a fallacy in the Text. The label can be 'Cherry Picking' and 'Vagueness' and 'Red Herring' and 'False Causality' and 'Irrelevant Authority' and 'Evading the burden of proof' and 'Strawman' and 'False Analogy' and 'Faulty Generalization'."
            "Please detect a fallacy in the Text.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the nine labels stated."
        )
        # system_message_zcot = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Cherry Picking' and 'Vagueness' and 'Red Herring' and 'False Causality' and 'Irrelevant Authority' and 'Evading the burden of proof' and 'Strawman' and 'False Analogy' and 'Faulty Generalization'.\n"
        #     "Please detect a fallacy in the Text.\nLet's think step by step.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the nine labels stated."
        # )
        # system_message_def = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Cherry Picking' and 'Vagueness' and 'Red Herring' and 'False Causality' and 'Irrelevant Authority' and 'Evading the burden of proof' and 'Strawman' and 'False Analogy' and 'Faulty Generalization'.\n"
        #     "1.Cherry Picking : The act of choosing among competing evidence that which supports a given position, ignoring or dismissing findings which do not support it.\n"
        #     "2. Vagueness : A word/a concept or a sentence structure which are ambiguous are shifted in meaning in the process of arguing or are left vague being potentially subject to skewed interpretations.\n"
        #     "3. Red Herring : The argument supporting the claim diverges the attention to issues which are irrelevant for the claim at hand.\n"
        #     "4. False Causality : X is identified as the cause of Y when another factor Z causes both X and Y OR X is considered the cause of Y when actually it is the opposite. It is assumed that because B happens after A, it happens because of A. In other words a causal relation is attributed where, instead, a simple correlation is at stake.\n"
        #     "5. Irrelevant Authority : An appeal to authority is made where the it lacks credibility or knowledge in the discussed matter or the authority is attributed a tweaked statement.\n"
        #     "6. Evading the burden of proof : A position is advanced without any support as if it was self-evident.\n"
        #     "7. Strawman : The arguer misinterprets an opponent’s argument for the purpose of more easily attacking it, demolishes the misinterpreted argument, and then proceeds to conclude that the opponent’s real argument has been demolished.\n"
        #     "8. False Analogy : because two things [or situations] are alike in one or more respects, they are necessarily alike in some other respect.\n"
        #     "9. Faulty Generalization : A generalization is drawn from a sample which is too small, not representative or not applicable to the situation if all the variables are taken into account.\n"
        #     "Please detect a fallacy in the Text."
        # )
        # system_message_query = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Cherry Picking' and 'Vagueness' and 'Red Herring' and 'False Causality' and 'Irrelevant Authority' and 'Evading the burden of proof' and 'Strawman' and 'False Analogy' and 'Faulty Generalization'."
        #     "Please detect a fallacy in the Text based on the Query.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the nine labels stated."
        # )
        # system_message_pr = (
        #     "Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\n"
        #     "The ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\n"
        #     "The label can be 'Cherry Picking' and 'Vagueness' and 'Red Herring' and 'False Causality' and 'Irrelevant Authority' and 'Evading the burden of proof' and 'Strawman' and 'False Analogy' and 'Faulty Generalization'.\n"
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
            if 'cherry picking' == sample[1]:
                ground_truth.append(1)
            elif 'vagueness' == sample[1]:
                ground_truth.append(2)
            elif 'red herring' == sample[1]:
                ground_truth.append(3)
            elif 'false causality' == sample[1]:
                ground_truth.append(4)
            elif 'irrelevant authority' == sample[1]:
                ground_truth.append(5)
            elif 'evading the burden of proof' == sample[1]:
                ground_truth.append(6)
            elif 'strawman' == sample[1]:
                ground_truth.append(7)
            elif 'false analogy' == sample[1]:
                ground_truth.append(8)
            elif 'faulty generalization' == sample[1]:
                ground_truth.append(9)
            
            user_message = f"{t1}\nLabel: "
            # user_message_def =f"{t1}\nLabel: "
            # user_message_zcot=f"{t1}\nPlease analyze step by step to detect any fallacy.\nLabel: "
            # user_message_query = f"{t1}\n{t5}\nLabel: "
            # user_message_pr = f"{t1}\n{t2}\n{t3}\n{t4}\n{t5}\nLabel: "
            pred = generate_response(system_message=system_message,user_message=user_message)
            # pred = generate_response(system_message=system_message_def,user_message=user_message_def)
            # pred = generate_response(system_message=system_message_zcot,user_message=user_message_zcot)
            # pred,response_ids,token_logprobs,total_logprob = generate_response1(system_message=system_message_query,user_message=user_message_query)
            # pred = generate_response(system_message=system_message_pr,user_message=user_message_pr)
            # Determine the label based on the first occurrence of keywords
            pred_lower = pred.lower()
            labels = [
                ('picking', 1),
                ('vagueness', 2),
                ('red herring', 3),
                ('causality', 4),
                ('cause', 4),
                ('authority', 5),
                ('evading', 6),
                ('strawman', 7),
                ('analogy', 8),
                ('generalization', 9)
            ]

            min_position = float('inf')
            selected_label = 0

            for label, code in labels:
                position = pred_lower.find(label)
                if 0 <= position < min_position:
                    min_position = position
                    selected_label = code
            
            # selected_label, selected_log_prob = get_label_log_prob(pred_lower, token_logprobs, labels)

            # gpt_preds.append(selected_label)
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
        
        
        # 계산된 정확도를 리스트에 추가
        accuracies = []
        for label in [1, 2,3,4,5,6,7,8,9]:
            true_labels = [1 if gt == label else 0 for gt in ground_truth]
            pred_labels = [1 if pred == label else 0 for pred in gpt_preds]
            acc = accuracy_score(true_labels, pred_labels)
            accuracies.append(acc)

        # 클래스별 정확도 출력
        print("Class 1 (Cherry Picking) Accuracy:", accuracies[0])
        print("Class 2 (Vagueness) Accuracy:", accuracies[1])
        print("Class 3 (Red Herring) Accuracy:", accuracies[2])
        print("Class 4 (False Causality) Accuracy:", accuracies[3])
        print("Class 5 (Irrelevant Authority) Accuracy:", accuracies[4])
        print("Class 6 (Evading the burden of proof) Accuracy:", accuracies[5])
        print("Class 7 (Strawman) Accuracy:", accuracies[6])
        print("Class 8 (False Analogy) Accuracy:", accuracies[7])
        print("Class 9 (Faulty Generalization) Accuracy:", accuracies[8])
        
        # 클래스별로 Precision, Recall, F1-Score 계산
        class_metrics = precision_recall_fscore_support(ground_truth, gpt_preds,average=None)

        print('class_metrics',class_metrics)
        # 클래스별 결과 출력
        classes = ["Cherry Picking","Vagueness","Red Herring","False Causality","Irrelevant Authority","Evading the burden of proof","Strawman","False Analogy","Faulty Generalization"]
        classes1 = ["The Other","Cherry Picking","Vagueness","Red Herring","False Causality","Irrelevant Authority","Evading the burden of proof","Strawman","False Analogy","Faulty Generalization"]
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
        
    # with open('./new_data/CLIMATE/climate_test.json','w') as f:

        
    #     json.dump(json_data, f, indent=4)  

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")