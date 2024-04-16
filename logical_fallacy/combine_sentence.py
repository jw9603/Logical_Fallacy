import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import sentiwordnet as swn
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def generate_combined_sentences(sentences, swapped_sentences):
    combined_sentences = []
    for i in range(len(sentences)):
        for j in range(len(swapped_sentences)):
            if i != j:  # 같은 위치에 있는 문장은 연결되지 않도록 함
                combined_sentence = sentences[i] + " , " + swapped_sentences[j] if i < j else swapped_sentences[j] + " , " + sentences[i]
                combined_sentences.append(combined_sentence)
    # Add combinations within swapped_sentences while preserving order
    combined_sentences.extend([swapped_sentences[i] + " , " + swapped_sentences[j] for i in range(len(swapped_sentences)) for j in range(i + 1, len(swapped_sentences))])
    return combined_sentences


if __name__ =='__main__':
    
    CALLS = 0
    
    with open('./data/new_total_split_v2_modified_negation_sum copy.json') as f:
        json_data = json.load(f)
        
        
    TOTAL_CALLS = len(json_data['test'])
    with open('./total_combined_negation.txt','w') as output_file:
        import sys
        original_stdout = sys.stdout
        # sys.stdout = output_file
        
        for sample in json_data['test']:
            texts = sample[2].split(',')
            print('sample[2]',sample[2])
            negation_text = sample[3]
            print('negation_text',negation_text)
            final_result = generate_combined_sentences(texts,negation_text)
            print('final_result',final_result)
            print(len(final_result))
            print(final_result[0])
            print(final_result[1])
            assert -1 == 0
            CALLS += 1
            print(CALLS, '/', TOTAL_CALLS)
        with open('./data/new_total_split_v2_modified_negation_sum copy.json','w') as f:
            json.dump(json_data,f,indent=4)
        
        print('키워드들이 JSON 파일에 저장되었습니다.')
        
        sys.stdout = original_stdout
    print("모든 출력이 'output.txt' 파일에 저장되었습니다.")