import pandas as pd
import ast
from datasets import load_dataset
from collections import defaultdict

class DataLoader:
    def __init__(self, go_dataset_path, dollarstreet_csv_path):
        self.go_dataset = load_dataset(go_dataset_path)
        self.dollarstreet_data = pd.read_csv(dollarstreet_csv_path)

    def get_wvs_data(self):
        train_data = self.go_dataset['train']
        train_data_wvs = train_data.filter(lambda x: x['source'] == 'WVS')
        questions = train_data_wvs['question']
        selections = train_data_wvs['selections']
        options = train_data_wvs['options']
        return questions, selections, options

    def get_dollarstreet_data(self):
        return self.dollarstreet_data

class DataProcessor:
    def __init__(self, wvs_selections):
        self.wvs_countries = self.extract_countries(wvs_selections)

    def extract_countries(self, wvs_selections):
        countries = []
        for response in wvs_selections:
            each_response = ast.literal_eval(response.strip("defaultdict(<class 'list'>, ").strip(")"))
            countries.extend(each_response.keys())
        return list(set(countries))

    def filter_common_countries(self, dollarstreet_countries):
        return list(set(self.wvs_countries).intersection(dollarstreet_countries))

    def prepare_family_data(self, dollarstreet_data, common_countries):
        return dollarstreet_data[
            dollarstreet_data['topics'].isin(['Family', 'Family snapshots']) &
            dollarstreet_data['country.name'].isin(common_countries) &
            dollarstreet_data['type'].isin(['image'])
        ]

