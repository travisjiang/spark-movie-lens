import os, math
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext, SparkConf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from itertools import combinations
import operator

import numpy as np


class ItemBasedCF:
    """A item recommendation engine
    * train
    * test
    * predict
    * recommend_for_user
    * update
    """

    def __init__(self, training_RDD, model_path):
        # train and save model
        if not os.path.exists(model_path):
            # Train the model
            self.rank = 8
            self.seed = 5L
            self.iterations = 10
            self.regularization_parameter = 0.1
            self._train_model()

            # Save model
            logger.info("ALS model saved to %s!" % model_path)
            self.model.save(self.sc, model_path)

        # load model
        else:
            logger.info("ALS model loaded from %s!" % model_path)
            self.model = MatrixFactorizationModel.load(sc, model_path)
            logger.info("ALS model loaded!")


    def train(self, rdd):
        """Train the ItemBasedCF with the current dataset
        """
        logger.info("Training the ItemBasedCF model...")
        self.item_sim_dict = _compute_similarity(rdd)
        logger.info("ALS model built!")

        #(user_id, [(item_id, rating)])
        user_items = rdd.map(lambda x: (x[0], (x[1], x[2])))\
                        .groupByKey().mapValues(list)

        #(user_id, [(unrated_item, rating)])
        self.item_pred_ratings = user_items.map(lambda x: (x[0], _compute_ratings(x[1], self.item_sim_dict)))

    def predictAll(self, rdd):
        """Predict with input rdd format (user_id, item_id)
        Returns rdd with format(user_id, item_id, rating)
        """
        pred_ratings = self.item_pred_ratings.flatMap(lambda x: _flat_grouped_item(x[0], x[1]))
            .join(rdd.map(lambda x:(x[0],x[1],1)))\
            .map(lambda x:(x[0]

    def _flat_grouped_item(user_id, items_with_rating):
        return [((user_id, t), r) for t, r in items_with_rating]

    @staticmethod
    def _compute_ratings(items_with_rating, sim_dict):
        """
        items_with_rating: [(item_id, rating)]
        sim_dict:          {item_id:[(item_id, sim)]}
        Returns:           [(unrated_item_id, rating)]
        """
        pred_ratings = {}
        sim_sums = {}
        rated_set = set([t for t,_ in items_with_rating])
        for rated_item, rating in items_with_rating:
            sim_items = sim_dict.get(rated_item, None)
            if sim_items:
                for sim_item, sim in sim_items:
                    if sim_item in rated_set:
                        continue
                    if not pred_ratings.has_key(sim_item):
                        pred_ratings[sim_item] = 0
                        sim_sums[sim_item] = 0
                    pred_ratings[sim_item] += sim * rating
                    sim_sums[sim_item] += sim
        pred_ratings = [(unrated_item, rating/sim_sums[unrated_item])\
            for unrated_item, rating in pred_ratings]

        return sorted(pred_ratings,key=operator.itemgetter(1),reverse=True)

    @staticmethod
    def _compute_similarity(rdd):
        """Compute sim using rdd with format(user_id, item_id, ratings)
        Returns dict with format{item_id: [(item_id, sim)]}
        """

        #(user_id, [(item_id, rating)])
        user_items = rdd.map(lambda x: (x[0], (x[1], x[2])))\
                        .groupByKey().mapValues(list)
        #([((item1, item2), (rating1, rating2))]
        item_pairs = user_items.flatMap(lambda x: _find_item_pairs(x[1]))

        #((item1, item2), [(rating1, rating2)])
        item_pairs = item_pairs.groupByKey().mapValues(list)

        #((item1, item2), sim)
        item_sims = item_pairs.map(lambda x: (x[0], _pearson_sim(x[1])))

        #(item1, [(item, sim)])
        item_sims = item_sims.flatMap(\
            lambda x:[(x[0][0], (x[0][1], x[1])), (x[0][1], (x[0][0], x[1]))])\
            .distinct().groupByKey().mapValues(list)

        item_sim_dict = {}
        for t, ts in item_sims.collect():
            item_sim_dict[t] = ts
        return item_sim_dict

    @staticmethod
    def _pearson_sim(ratings):
        """pearson similarity"""
        ratings1, ratings2 = zip(*ratings)
        x, y= np.array(ratings1), np.array(ratings2)
        up = sum((x-x.mean())*(y-y.mean()))
        down = math.sqrt(sum(x**2)*sum(y**2))
        return up/down

    @staticmethod
    def _find_item_pairs(item_with_rating):
        return [((item1[0], item2[0]),(item1[1], item2[1]))\
            for item1, item2 in combinations(item_with_rating, 2)]

