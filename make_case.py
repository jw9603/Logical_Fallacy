from openai import OpenAI
import json
from retry import retry
import time

@retry()
def generate_counterarg(text,fallacy_class,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. The text contains one of following logical fallacies:{fallacy_class}. Represent the counterargument to the text.\n\nText: {text}\nCounterargument:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()
@retry()
def generate_counterarg_cbr(text,client):
    prompt = f"Represent the counterargument to the text.\n\nText: {text}\nCounterargument:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()


@retry()
def generate_explanation(text,fallacy_class,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. The text contains one of following logical fallacies:{fallacy_class}. Analyze the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()

@retry()
def generate_explanation_cbr(text,client):
    prompt = f"Analyze the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()

@retry()
def generate_goal(text,fallacy_class,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. The text contains one of following logical fallacies:{fallacy_class}. The text contains a logical fallacy. Express the goal of the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()

@retry()
def generate_goal_cbr(text,client):
    prompt = f"Express the goal of the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1.0,
        top_p=0.1
    )
    return response.choices[0].text.strip()



def main():
    # OpenAI 클래스의 인스턴스 생성
    CALLS = 0
    client = OpenAI(
        api_key="YOURKEY"
    )


    #### It is for each dataset's class #######
    argotario_class = ['Appeal to Emotion','Faulty Generalization','Red Herring','Ad Hominem', 'Irrelevant Authority']
    logic_class = ['Faulty Generalization','Ad Hominem','False Causality','Ad Populum','Circular Reasoning','Appeal to Emotion','Deductive Reasoning','Red Herring','Intentional Fallacy','False Dilemma','Irrelevant Authority','Fallacy of Extension','Equivocation']
    climate_covid_class = ['Cherry picking','Vagueness','Red Herring','False Causality','Irrelevant Authority','Evading the burden of Proof','Strawman','False Analogy','Faulty Generalization']
    propaganda_class=['loaded language','exaggeration or minimisation','doubt','strawman','flag waving','thought terminating cliches','appeal to fear','name calling','whatboutism','false causality','irrelevant authority','slogans','reductio ad hitlerum','red herring','black and white fallacy']


    with open('./new_data/COVID-19/covid_dev.json') as f: # The data you will generate  contextual augmentations

        json_data = json.load(f)
    results = []
  
    TOTAL_CALLS = len(json_data['test'])
    
    with open('./query/COVID-19/dev_counterarg.txt','w') as output_file: # The folder for saving the generated contextual augmentations


        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
    
        for sample in json_data['test']:
            text = sample[0]
            fallacy_type = ', '.join(climate_covid_class)
       
            
            try:
                result = generate_counterarg(text,fallacy_type,client)

         
             
            except Exception as e:
                print(f"API 호출 중 오류 발생: {e}")
                continue
            
            print('result:',result )
            results.append(result)
            
            sample[2] = result
            CALLS += 1
            print(CALLS, '/', TOTAL_CALLS)
    
        with open('./new_data/COVID-19/covid_dev.json','w') as f:
        
            json.dump(json_data, f, indent=4)
        
        print('질문이 JSON 파일에 저장되었습니다.')
        
        sys.stdout = original_stdout
    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()

    

