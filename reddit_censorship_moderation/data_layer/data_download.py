import datetime
import logging
import os
import time
from calendar import calendar
import datetime
import asyncio
# import json
import pickle
import string
import sys
import time
import logging

import numpy as np
from dotenv import load_dotenv
from requests.exceptions import ChunkedEncodingError
from mongoTools import *
from pmaw import PushshiftAPI
import prawcore
import pymongo
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


def convert_time_format(comment_or_post):
    comment_or_post['created_utc'] = datetime.datetime.fromtimestamp(
        comment_or_post['created_utc']).isoformat().split(
        "T")


class data_download:
    def __init__(self):
        self.chunk_counter = None
        self.chunk_size = None
        self.curr_tmp_chunk = None
        self.curr_chunk = None
        self.times_praw = [0, 0]
        self.times_psaw = [0, 0]
        self.psaw = PushshiftAPI(num_workers=12)
        self.praw = praw.Reddit(
            client_id=os.getenv('CLIENT_ID'),
            client_secret=os.getenv('CLIENT_SECRET'),
            user_agent=os.getenv('USER_AGENT'),
            username=os.getenv('USER_NAME'),
            password=os.getenv('PASSWORD'),
            check_for_async=False
        )
        self.download_func = {'post': self.pushift.search_submissions, 'comment': self.pushift.search_comments}

    def handle_single_submission(self, sub, kind, _post_or_comment, pbar_):
        s_time = time.time()
        # try:
        if '_reddit' in sub.keys():
            del sub["_reddit"]
        if 'subreddit' in sub.keys():
            del sub["subreddit"]
        if 'author' in sub.keys():
            del sub["author"]  # TODO keep author (check)
        if 'poll_data' in sub.keys():
            sub['poll_data'] = str(sub['poll_data'])
        convert_time_format(sub)
        post_id = sub["id"]
        if kind == "reddit_api":
            self.curr_tmp_chunk[post_id] = {}
            self.curr_tmp_chunk[post_id]["post_id"] = post_id
            sub = dict(sorted(sub.items(), key=lambda item: item[0]))
            self.curr_tmp_chunk[post_id][kind] = sub
            e_time = time.time()
            self.times_praw[0] += (e_time - s_time)
            self.times_praw[1] += 1
        else:
            if post_id in self.curr_tmp_chunk:
                for k in sub.copy():
                    if k in self.curr_tmp_chunk[post_id]["reddit_api"]:
                        if sub[k] == self.curr_tmp_chunk[post_id]["reddit_api"][k]:
                            del sub[k]
            else:
                self.curr_tmp_chunk[post_id] = {}
                self.curr_tmp_chunk[post_id]["post_id"] = post_id
            self.curr_tmp_chunk[post_id][kind] = sub
            e_time = time.time()
            self.times_psaw[0] += (e_time - s_time)
            self.times_psaw[1] += 1
            if len(self.curr_tmp_chunk[post_id]) == 3:
                self.chunk_counter[post_id] = self.curr_tmp_chunk[post_id].copy()
                del self.curr_tmp_chunk[post_id]
                self.chunk_counter -= 1
        pbar_.update(1)

        if self.chunk_counter <= 0:
            self.chunk_counter = self.chunk_size
            start_time_dump = time.time()
            await self.dump_data()
            pbar_.reset()
            logging.info("\nwrite to json time: {}.".format(time.time() - start_time_dump))

    def dump_data(self):
        pass

    async def run(self, _submissions_list, _submissions_list_praw, _post_or_comment):
        with tqdm(total=max(len(_submissions_list), len(_submissions_list_praw))) as pbar:
            await asyncio.gather(
                *[self.handle_single_submission(submission, 'reddit_api', _post_or_comment, pbar) for submission in
                  _submissions_list_praw])
            await asyncio.gather(
                *[self.handle_single_submission(submission, 'pushift_api', _post_or_comment, pbar) for submission in
                  _submissions_list])

    def download_operate(self, subreddit_name, year, submission_kind_list, start_day=1, start_month=1, m_step=1, d_step=1, run_type='m'):
        # args: 1: subredits_list, 2:year, 3: post_or_comment_lst(d), 4: start_day(d), 5: start_month(d), 6:env_num(d)

        data = {}
        counter = 0

        # parameters
        # load_dotenv(f"../../code_shimon/.env{args[6]}")
        # print(f"../../code_shimon/.env{args[6]}")
        # data_layer = DataLayer(os.getenv("AUTH_DB"))
        # if subreddit_name == 'update_all':
        #     sub_reddits = [x.split('_')[0] for x in data_layer.get_db_collections_names(year)][::-1]
        # else:
        #     sub_reddits = args[1].split(',')
        # post_or_comment_lst = args[3].split(',')
        # # for month in tqdm(range(12, 13, 1)):
        # # for day in tqdm(calendar.monthrange(year, month)):
        # # logging.info("month: {}".format(month))
        # for sub_reddit in sub_reddits:
        #     print(sub_reddit)
        for sub_kind in submission_kind_list:
            logging.info(f"Downloading {sub_kind}s")
            # TODO separate the data access
            # collection_name = f"{subreddit_name}_{sub_kind}"
            # print(collection_name)
            # mycol = data_layer.get_collection(year, subreddit_name, sub_kind)
            # index_name = 'pid'
            # if index_name not in mycol.index_information():
            #     mycol.create_index([('post_id', 1)], name=index_name, unique=True)
            # file_name = "{}_{}_{}.json".format(collection_name, year, sub_kind)
            # if args[1] == 'update_all' and year == 2022:
            #     max_date = list(
            #         mycol.find({}, {'reddit_api.created_utc': 1}).sort("reddit_api.created_utc.0", -1).limit(1))
            #     print(max_date)
            #     if len(max_date) > 0:
            #         max_date = max_date[0]['reddit_api']['created_utc'][0]
            #         start_month = int(max_date.split('-')[1])
            #     else:
            #         start_month = 1
            # ########
            # run_type = "monthly"
            # if post_or_comment == "comment":
            #     run_type = "dayly"

            for m in range(start_month, 13, m_step):
                last_day_of_month = calendar.monthrange(year, m)[1]
                logging.info(f'last day of month {m} is {last_day_of_month}')
                first_day = 1
                if m == start_month:
                    first_day = start_day
                for d in range(first_day, last_day_of_month + 1, d_step):
                    loop = self.download(d, m, sub_kind, subreddit_name, year)
                    if run_type == "m":
                        break
            # empty chunk
            if len(self.curr_chunk) > 0:
                s_time_dump = time.time()
                loop.run_until_complete((self.dump_data()))
                logging.info("Last write to json time: {}.".format(time.time() - s_time_dump))
        # client = pymongo.MongoClient(
        #     'mongodb://132.72.66.126:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl'
        #     '=false')
        # add_status(client[f"reddit_{year}"], f"{subreddit_name}_post")

    def download(self, d, m, year, sub_kind, subreddit_name):
        start_time = int(datetime.datetime(year, m, d, 0, 0).timestamp())
        end_time = int(datetime.datetime(year, m, d, 23, 59).timestamp())
        logging.info(f"start date:{d}/{m}/{year}")
        submissions_list_psaw = []
        submissions_list_praw = []
        start_run_time = time.time()
        try:
            submissions_list_psaw = self.download_func.get(sub_kind)(subreddit=subreddit_name,
                                                                     after=start_time,
                                                                     before=end_time, safe_exit=True)
            end_run_time = time.time()
            logging.info("Extract from pushift time: {}".format(end_run_time - start_run_time))
            self.change_reddit_mode()
            submissions_list_praw = self.download_func.get(sub_kind)(subreddit=subreddit_name,
                                                                     after=start_time,
                                                                     before=end_time)
            self.change_reddit_mode()
            logging.info("Extract from reddit time: {}".format(time.time() - end_run_time))
        except ChunkedEncodingError as e:
            logging.warn(f"Error at {d}/{m}/{year}")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.run(submissions_list_psaw, submissions_list_praw, sub_kind))
        # asyncio.run(main(submissions_list, data, counter))
        if len(self.curr_tmp_chunk) > 0:
            self.curr_chunk.update(self.curr_tmp_chunk)
            self.curr_tmp_chunk = {}
        logging.info(f"Mean time to handle reddit {sub_kind} is: {self.times_praw[0] / self.times_praw[1]}")
        logging.info(
            f"Mean time to handle pushift {sub_kind} is: {self.times_praw[0] / self.times_praw[1]}")
        return loop

    def change_reddit_mode(self):
        if self.psaw.praw:
            self.psaw.praw = None
        else:
            self.psaw.praw = self.praw
