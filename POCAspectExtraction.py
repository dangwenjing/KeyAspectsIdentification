import os
import random
import shutil
from transformers import pipeline
import os
import json
import numpy as np


def read():
    rootpath = 'demo_data'
    filepath_list = []
    for x in os.listdir(rootpath):
        filepath = os.path.join(rootpath, x)
        if "aspects" not in filepath:
            filepath_list.append(filepath)
            print(filepath)
    return filepath_list


def extract(filepath_list):

    label_names = ['O', 'B-title', 'I-title', 'B-author', 'I-author', 'B-time', 'I-time','B-reference', 'B-plat', 'I-plat','B-version', 'I-version']
    #  Replace this with your own checkpoint
    # model_checkpoint = 'store_model/bert-base-uncased-finetuned-ner-117_alldata/checkpoint-38000'
    model_checkpoint = 'trained_model'

    token_classifier = pipeline(
        "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    )

    sumnum = len(filepath_list)
    countnum = 0
    for filepath in filepath_list:

        countnum += 1
        print(f'log: {countnum}/{sumnum}', end='\r')

        with open(filepath, 'r', encoding='utf-8') as file:
            file_content = file.read()

        results = token_classifier(file_content)

        for item in results:
            item['entity_group'] = label_names[int(item['entity_group'].split('_')[1])]

        final_results = []
        pre_label = ""
        for result in results:
            label = result['entity_group']
            if label == 'O':
                continue
            ordlabel = label.split('-')[0]
            nerlabel = label.split('-')[1]
            if nerlabel !=  pre_label:
                result['entity_group'] = nerlabel
                final_results.append(result)
            elif ordlabel == 'I':
                final_results[-1]['word'] += ' '+result['word']
                final_results[-1]['end'] = result['end']
            pre_label = nerlabel

        for result in final_results:
            for key, value in result.items():
                if isinstance(value, np.float32):
                    result[key] = value.item()

        directory, filename = os.path.split(filepath)
        print(os.path.join(directory, filename+'_aspects.json'))
        with open(os.path.join(directory, filename+'aspects.json'), "w") as file:
            json.dump(final_results, file, indent=4)

if __name__ == "__main__":
    filepath_list = read()
    extract(filepath_list)