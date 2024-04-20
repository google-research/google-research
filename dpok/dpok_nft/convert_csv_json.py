import pandas as pd
import argparse
import json


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--csv_dir", type=str)
    
    return arg_parser.parse_args()

def _convert(args):
    json_dict = {}
    df = pd.read_csv(args.csv_dir)
    print(len(df))
    for i, r in df.iterrows():
        print(i)
        print(r['file_name'])
        json_dict[r['text']] = {'category': None}
    with open("../dataset/drawbench/data_meta_new.json", 'w', encoding='utf-8') as json_file:
        json.dump(json_dict, json_file, ensure_ascii=False, indent=4)
    _sanity_check(df, json_dict)
    
def _sanity_check(df: pd.DataFrame, json_dict: dict):
    text_set_df = set(df['text'].unique())
    print(len(text_set_df))
    text_set_dict = set(json_dict.keys())
    print(len(text_set_dict))
    missing_texts = text_set_df - text_set_dict

    print(missing_texts)
    print("Missing:")
    for text in missing_texts:
        print(text)
if '__main__' == __name__:
    args = parse_args()
    _convert(args)