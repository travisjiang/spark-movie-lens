import os, math
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext, SparkConf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from itertools import combinations
import operator


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




    def _compute_similarity(self, rdd):
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
        item_sims = item_pairs.map(lambda x: (x[0], _calc_item_sim(x[1])))

    def _calc_item_sim(ratings):
        ratings_sum = reduce(lambda x,y:(x[0]+y[0], x[1]+y[1]), ratings)
        ratings_mean = reduce(lambda x,y:(x/len(ratings), y/len(ratings)))



    def _find_item_pairs(item_with_rating):
        return [((item1[0], item2[0]),(item1[1], item2[1]))\
            for item1, item2 in combinations(item_with_rating, 2)]


    def compute_ratings(user_id, items_with_rating, sim_dict):
        """
        items_with_rate: [(item_id, rating)]
        sim_dict:{item_id:[(item_id, sim)]}
        Returns: [(unrated_item_id, rating)]
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

    def train(self, training_RDD):
        """Train the ItemBasedCF with the current dataset
        """
        logger.info("Training the ItemBasedCF model...")
        self._compute_similarity(training_RDD)
        logger.info("ALS model built!")

        #(user_id, [(item_id, rating)])
        user_items = self.training_RDD.map(lambda x: (x[0], (x[1], x[2])))\
                        .groupByKey().mapValues(list)

        #(user_id, [(unrated_item, rating)])
        self.pred_ratings = user_items.map(lambda x: (x[0], compute_ratings(x[1], self.sim_dict)))




    def predict_ratings(self, user_and_item_RDD):
        """Gets predictions for a given (user_id, item_id) formatted RDD
        Returns: an RDD with format (user_id, item_id, rating)
        """

        predicted_RDD = self.model.predictAll(user_and_item_RDD)

        return predicted_RDD


