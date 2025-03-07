import random
from openai import OpenAI
import json
from retry import retry
import time

@retry()
def generate_counterarg_query(text,cg,client):
    prompt = f"I'll give you some texts and text's counterarguments. The texts can be question and answer pairs or sentences. Create one question for each text that analyze the text based on counterarguments rather than directly asking what a logical fallacy is.\n\nText: {text}\nCounterargument: {cg}\nQuestion:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # You can set the maximum length of the question.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()

@retry()
def generate_explanation_query(text,ex,client):
    prompt = f"I'll give you some texts and text's explanations. The texts can be question and answer pairs or sentences. Create one question for each text that analyze the text based on explanations rather than directly asking what a logical fallacy is.\n\nText: {text}\nExplanation: {ex}\nQuestion:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # You can set the maximum length of the question.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()

@retry()
def generate_goal_query(text,goal,client):
    prompt = f"I'll give you some texts and text's goals. The texts can be question and answer pairs or sentences. Create one question for each text that analyze the text based on goals rather than directly asking what a logical fallacy is.\n\nText: {text}\nGoal:{goal}\nQuestion:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # You can set the maximum length of the question.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()


def main():
    # Create an instance of the OpenAI class
    CALLS = 0
    client = OpenAI(
        api_key="YOURKEY"
    )
    
    
    with open('./new_data/propaganda/propaganda_train.json') as f:
    
        json_data = json.load(f)
    questions = []
    TOTAL_CALLS = len(json_data['train'])
    with open('./query/propaganda/train_goal_query.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
    
        for sample in json_data['train']:
            text = sample[0]
            ex = sample[4]            
            try:
              
                question = generate_goal_query(text,ex,client)
             
            except Exception as e:
                print(f"An error occurred during the API call: {e}")
                continue
            
            print('query:',question)
            questions.append(question)
            
            sample[7] = question
            CALLS += 1
            print(CALLS, '/', TOTAL_CALLS)
    
        with open('./new_data/propaganda/propaganda_train.json','w') as f:
        
            json.dump(json_data, f, indent=4)
        
        print("The question has been saved to a JSON file.")
        
        sys.stdout = original_stdout
    print("All output has been saved to 'output.txt' file.")

if __name__ == "__main__":
    main()
