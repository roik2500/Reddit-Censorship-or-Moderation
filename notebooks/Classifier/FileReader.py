# from psaw import PushshiftAPI
# from pmaw import PushshiftAPI as PushshiftApiPmaw
from datetime import datetime
import requests
import json
import pandas as pd
import ijson
from tqdm import tqdm
import pymongo
import os
import csv


class FileReader:

    def __init__(self):
        self.json_open_file = None

    ''' :return dict '''
    def read_from_json_to_dict(self, PATH):
        with open(PATH) as json_file:
            data = json.load(json_file)
        return data

    ''' :return data frame '''
    def read_from_json_to_df(self, PATH):
        df = pd.read_json(PATH, lines=True)
        return df

    def read_from_csv(self, path):
        df = pd.read_csv(path)
        return df

    def write_to_csv(self, path, file_name, df_to_write):
        if path[-1:] == '/':
            df_to_write.to_csv(path + file_name)
        else:
            df_to_write.to_csv(path + '/' + file_name)

    def write_dict_to_json(self, path, file_name, dict_to_write):
        file = path + file_name + '.json'
        with open(file, 'w') as fp:
            json.dump(dict_to_write, fp)

    def get_specific_items_by_post_ids_from_json(self, ids_list):
        text_and_date_list = []
        with open(os.getenv('DATA_PATH')) as json_file:
            data = json.load(json_file)
            for key_id in ids_list:
                post = data[key_id]
                text_and_date_list.append(self.get_text_from_post_OR_comment(post, post_or_comment='post'))
        return text_and_date_list  # [title , selftext ,created_utc, 'id']

    '''
    @:argument json_file - path to the file to read from
    @:return return json iterator to the file.
    '''
    def get_json_iterator(self, json_file):
        if self.json_open_file != None and self.json_open_file.open:
            self.json_open_file.closed()
        self.json_open_file = open(json_file, 'rb')
        items = ijson.items(self.json_open_file, 'item')
        return items


    def topic_number_connected_posts(self, path, folder_to_save, number_of_topic=20):
        topic_df = self.read_from_csv(path)
        topic_num_posts_dict = {}
        for topic_number in range(number_of_topic):
            topic_num_posts_dict[topic_number] = topic_df.loc[topic_df['Dominant_Topic'] == topic_number]['post_id'].to_list()
        return topic_num_posts_dict


    def write_dict_to_csv(self, file_name, dictionary):
        with open(file_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dictionary.items():
               writer.writerow([key, value])

    def read_dict_from_csv(self, file_name):
        with open(file_name) as csv_file:
            reader = csv.reader(csv_file)
            mydict = dict(reader)
        return mydict

    def testing_recorder(self, features_data, list_of_data_to_record_train, list_of_data_to_record_valid, list_of_data_to_record_test,
                         classifier_data, max_depth="--",
                         csv_record_path=f"G:\.shortcut-targets-by-id\1lJuBfy-iW6jibopA67C65lpds3B1Topb\Reddit Censorship Analysis\final_project\Features\testing\test_xl.csv"):
        ID = pd.read_csv(csv_record_path).tail(1).values.tolist()[0]
        ID = [ID[0]+1]
        now = datetime.now() # current date and time
        time = now.strftime("%H:%M:%S")
        exec_date = now.strftime("%m/%d/%Y")
        date_time = [exec_date, time]
        max_depth_list = [max_depth]
        data = ID + date_time + max_depth_list + classifier_data + features_data + list_of_data_to_record_train + list_of_data_to_record_valid +list_of_data_to_record_test
        print("classifier_data: ", data)
        with open(csv_record_path, 'a', newline='') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(data)
            f_object.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'C:\\Users\\User\\Documents\\FourthYear\\Project\\document_topic_table_general-1_updated\\document_topic_table_general-1_updated.csv'
    file_reader = FileReader()
    file_reader.topic_number_connected_posts(path)
