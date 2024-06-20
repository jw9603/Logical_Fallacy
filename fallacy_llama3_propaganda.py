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
import torch.nn.functional as F

os.environ['HF_TOKEN'] = 'YOURKEY'

# Initialize the model and tokenizer
# model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the correct model path if different
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

def generate_response1(system_message, user_message): # This function is used to calculate the confidence score.
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
    
    with open('./new_data/propaganda/propaganda_test_llama.json') as f:
        json_data = json.load(f)
        
    TOTAL_CALLS = len(json_data['test'])
    ranking_prompt = ['Counterargument Query','Explanation Query','Goal Query']
    # with open('./result/propaganda/llama2/llama2_no_def_7B.txt','w') as output_file:
    with open('./result/propaganda/llama2/llama2_promptranking1_13B.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
        
        system_message = (
            "Your task is to detect a fallacy in the text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n"
            "Please detect a fallacy in the Text.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the 15 labels stated."
        )
        # system_message_zcot = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n"
        #     "Please detect a fallacy in the Text.\nLet's think step by step.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the 15 labels stated."
        # )
        # system_message_def = (
        #     "Your task is to detect a fallacy in the Text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n"
        #     "1. Loaded Language : Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.\n"
        #     "2. Exaggeration or Minimisation : Either representing something in an excessive manner: making things larger, worse or making something seem less important than it really is.\n"
        #     "3. Doubt : Questioning the credibility of someone or something.\n"
        #     "4. Strawman : When an opponent’s proposition is substituted with a similar one which is then refuted in place of the original proposition.\n"
        #     "5. Flag Waving : Playing on strong national feeling (or to any group) to justify/promote an action/idea.\n"
        #     "6. Thought-Terminating Cliches : Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or distract attention away from other lines of thought.\n"
        #     "7. Appeal to Fear: Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases the support is based on preconceived judgements.\n"
        #     "8. Name Calling: Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable or loves, praises.\n"
        #     "9. Whatboutism: A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument.\n"
        #     "10. False Causality: Assuming a single cause or reason when there are actually multiple causes for an issue.\n"
        #     "11. Irrelevant Authority: Stating that a claim is true simply because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. We consider the special case in which the reference is not an authority or an expert in this technique, although it is referred to as Testimonial in literature.\n"
        #     "12. Slogans: A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.\n"
        #     "13. Reductio Ad Hitlerum:Persuading an audience to disapprove an action or idea by suggesting that the idea is popular with groups hated in contempt by the target audience. It can refer to any person or concept with a negative connotation.\n"
        #     "14. Red Herring: Introducing irrelevant material to the issue being discussed, so that everyone’s attention is diverted away from the points made.\n"
        #     "15. Black and White Fallacy: Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an the extreme case, tell the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).\n"
        #     "Please detect a fallacy in the Text."
        # )
        # system_message_query = (
        #     "Your task is to detect a fallacy in the text. The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n"
        #     "Please detect a fallacy in the Text based on the Query.\nReturn only the name of the label, and nothing else. MAKE SURE your output is one of the 15 labels stated."
        # )
        # system_message_pr = (
        #     "Given a Sentence with a logical fallacy, we aim to detect it using queries based on multiple perspectives, such as counterargument, explanation, and goal.\n"
        #     "The ranking prompt indicates the order of queries based on their confidence scores, which are helpful in identifying the specific type of logical fallacy present in the sentence.\n"
        #     "The label can be 'Loaded Language' and 'Exaggeration or Minimisation' and 'Doubt' and 'Strawman' and 'Flag Waving' and 'Thought terminating cliches' and 'Appeal to Fear' and 'Name Calling' and 'Whatboutism' and 'False Causality' and 'Irrelevant Authority' and 'Slogans' and 'Reductio Ad Hitlerum' and 'Red Herring' and 'Black and White Fallacy'.\n"
        #     "Based on the ranking prompt of these queries, please reference them to detect the fallacy in the sentence."
        # )
        
        for sample in json_data['test']:
            # These are the parameters to be used for running each of the three queries individually, or for Zcot and DEF.
            t1 = 'Text:' +sample[0]
            # t3 = 'Query: '+sample[5] # CG
            # t4 = 'Query: '+sample[6] # EX
            # t5 = 'Query: '+sample[7] # GO
            
            
            # These are the parameters to be used for running each of the prompt ranking
            # t1 = 'Sentence:'+sample[0]
            
            # score_cg = sample[9]
            # score_ex = sample[10]
            # score_go = sample[11]
            # t2 = rank_confidence_scores(score_cg,score_ex,score_go,ranking_prompt)
            # # t2 = 'Ranking Prompt:' +', '.join(sample[15])
            # t3 = 'Counterargument Query: '+sample[5] # CG
            # t4 = 'Explanation Query: '+sample[6] # EX
            # t5 = 'Goal Query: '+sample[7] # GO
            
            # t2 = 'Ranking Prompt:' +', '.join(t2)
            
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
            
            user_message = f"{t1}\nLabel: "
            # user_message_def =f"{t1}\nLabel: "
            # user_message_zcot=f"{t1}\nPlease analyze step by step to detect any fallacy.\nLabel: "
            # user_message_query = f"{t1}\n{t5}\nLabel: "
            # user_message_pr = f"{t1}\n{t2}\n{t3}\n{t4}\n{t5}\nLabel: "
            pred= generate_response(system_message=system_message,user_message=user_message)
            # print('proba',probabilities)
            # print('response',response_ids)
            # pred = generate_response(system_message=system_message_def,user_message=user_message_def)
            # pred = generate_response(system_message=system_message_zcot,user_message=user_message_zcot)
            # pred = generate_response(system_message=system_message_query,user_message=user_message_query)
            # pred,response_ids,token_logprobs,total_logprob = generate_response1(system_message=system_message_query,user_message=user_message_query)
            # pred = generate_response(system_message=system_message_pr,user_message=user_message_pr)
            # Determine the label based on the first occurrence of keywords
            pred_lower = pred.lower()
            labels = [
                ('loaded language', 1),
                ('exaggeration or minimisation', 2),
                ('doubt', 3),
                ('strawman', 4),
                ('flag waving', 5),
                ('thought terminating cliches', 6),
                ('appeal to fear', 7),
                ('name calling', 8),
                ('whataboutism', 9),
                ('false causality', 10),
                ('irrelevant authority', 11),
                ('slogans', 12),
                ('reductio ad hitlerum', 13),
                ('red herring', 14),
                ('black and white fallacy', 15)
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
        
        
    # with open('./new_data/propaganda/propaganda_test_llama.json','w') as f:

        
    #     json.dump(json_data, f, indent=4)    

    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")