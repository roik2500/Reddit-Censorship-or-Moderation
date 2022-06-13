import os

import pymongo
import logging

logging.basicConfig(format='%(asctime)s %(message)s')


class DataLayer:
    @staticmethod
    def get_author_fullname(submission):
        return submission["author_fullname"]

    @staticmethod
    def get_num_comments(submission):
        return submission["reddit_api"]["num_comments"]

    @staticmethod
    def get_post_id(submission):
        return submission["post_id"]

    @staticmethod
    def get_retrieved(submission):
        return submission["retrieved"]

    @staticmethod
    def get_status(submission):
        return submission["status"]

    @staticmethod
    def get_link_flair(submission):
        try:
            return submission["reddit_api"]["link_flair_text"]
        except KeyError as e:
            return ''

    @staticmethod
    def get_date(submission):
        return submission["reddit_api"]["created_utc"]

    @staticmethod
    def get_title(submission):
        try:
            title_bool = submission["reddit_api"]['title'] == "[deleted by user]" or submission["reddit_api"][
                'title'] == "[deleted]" or submission["reddit_api"]['title'] == "[removed]"
            if title_bool and "pushift_api" in submission:
                sub_submission = submission["pushift_api"]
            else:
                sub_submission = submission["reddit_api"]
        except KeyError as e:
            if e == "reddit_api":
                sub_submission = submission["pushift_api"]
            else:
                return ''
        if 'title' in sub_submission:
            return sub_submission['title']
        else:
            # self.logger.warning(f"{self.get_post_id(submission)} is a comment and does not contains title")
            return ''

    @staticmethod
    def get_selftext(sub_kind, submission):
        sub_submission = ''
        if sub_kind == 'post':
            selftext = "selftext"
        else:
            selftext = "body"
        if "pushift_api" in submission:
            if selftext in submission["pushift_api"]:
                sub_submission = submission["pushift_api"]
            else:
                # self.logger.warning(f"{self.get_post_id(submission)} has the same text in pushift and reddit")
                sub_submission = submission["reddit_api"]
        else:
            # self.logger.warning(f"{self.get_post_id(submission)} does not contains pushift_api field. you got
            # reddit_api selftext")
            sub_submission = submission["reddit_api"]
        return sub_submission[selftext]

    def insert_many(self):
        pass
