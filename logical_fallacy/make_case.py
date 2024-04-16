import random
from openai import OpenAI
import json
from retry import retry
import time

random.seed(0)




@retry()
def generate_counterarg(text,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. Represent the counterargument to the text.\n\nText: {text}\nCounterargument:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1,
        top_p=1
    )
    return response.choices[0].text.strip()

@retry()
def generate_explanation(text,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. Analyze the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1,
        top_p=1
    )
    return response.choices[0].text.strip()

@retry()
def generate_goal(text,client):
    prompt = f"I'll give you some texts. The texts can be question and answer pairs or sentences. Express the goal of the text.\n\nText: {text}\nExplanation:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256,  # 질문의 최대 길이를 설정할 수 있습니다.
        temperature=1,
        top_p=1
    )
    return response.choices[0].text.strip()


def main():
    # OpenAI 클래스의 인스턴스 생성
    CALLS = 0
    client = OpenAI(
        api_key='YOURKEY'
    )
    text = "Annie must like Starbucks because all girls like Starbucks."
    print(generate_counterarg(text,client))
    print('\n')
    print(generate_explanation(text,client))
    print('\n')
    print(generate_goal(text,client))
    with open('./new_data/CLIMATE/climate_test_no_fallacy.json') as f:
    
        json_data = json.load(f)
    results = []
    TOTAL_CALLS = len(json_data['test'])
    
    # with open('./query/CLIMATE/test_counterargument_no_fallacy.txt','w') as output_file:
    # with open('./query/CLIMATE/test_explanation_no_fallacy.txt','w') as output_file:
    with open('./query/CLIMATE/test_goal_no_fallacy.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = output_file
    
        for sample in json_data['test']:
            text = sample[0]
            # fallacy_type = sample[1]
            # print('text',text)
            # print(type(text))
            
            try:
              
                result = generate_goal(text,client)
             
            except Exception as e:
                print(f"API 호출 중 오류 발생: {e}")
                continue
            
            print('result:',result )
            results.append(result)
            
            sample[4] = result
            CALLS += 1
            print(CALLS, '/', TOTAL_CALLS)
    
        with open('./new_data/CLIMATE/climate_test_no_fallacy.json','w') as f:

        
            json.dump(json_data, f, indent=4)
        
        print('질문이 JSON 파일에 저장되었습니다.')
        
        sys.stdout = original_stdout
    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()

    

