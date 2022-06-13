
import pymongo
import logging

from reddit_censorship_moderation.data_layer.data_layer import DataLayer

logging.basicConfig(format='%(asctime)s %(message)s')


class JsonDataLayer(DataLayer):
    def __init__(self):
        super(JsonDataLayer, self).__init__()
