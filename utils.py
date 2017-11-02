import os
import math
import operator
from operator import add
import logging

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext, SparkConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("item_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])

    return sc


def top_k(self, l, k, index):
    return sorted(l, key=operator.itemgetter(index), reverse=True)[:k]


def test_model_rmse(model, test_RDD):
    '''Test model by: rmse
    Args:
        model: Recommend model, must implement predictAll(rdd)
        k: Recommend top k items for each user
        training_RDD: Traning dataset used
        test_RDD: Test dataset to evaluate model
    Returns:

    '''

    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    predictions = model.predictAll(
        test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(
        lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(
        rates_and_preds.map(
            lambda r: (
                r[1][0] -
                r[1][1]) ** 2).mean())

    print 'For testing data the RMSE is %s' % (error)


def evaluate_model(self, model, k, training_RDD, test_RDD):
    '''Evaluate model with: precision, recall, coverage and popularity

    Args:
        model: Recommend model, must implement predictAll(rdd)
        k: Recommend top k items for each user
        training_RDD: Traning dataset used
        test_RDD: Test dataset to evaluate model
    Returns:
        (precison, recall, coverage, popularity)

    '''

    # 1. transfer test rdd
    # (user_id, [(item_id, rating)])
    user_item_test = test_RDD.map(lambda x: (x[0], (x[1], x[2]))) \
        .groupByKey().mapValues(list) \
        .map(lambda x: (x[0], top_k(x[1], k, 1)))
    # (user_id, [item_id])
    user_item_test = user_item_test.map(lambda x: (x[0],
                                                   [t for t, _ in x[1]]))
    print("user_item_test ", user_item_test.take(1))

    # 2. predict ratings
    # (user_id, [(item_id, rating)])
    pred_ratings = self.model.predictAll(test_RDD.map(lambda x: (x[0], x[1]))) \
        .map(lambda x: (x[0], (x[1], x[2]))) \
        .groupByKey().mapValues(list)
    print("pred_ratings", pred_ratings.take(1))

    # 3. get top k recommend item
    # (user_id, [(item_id, rating)])
    user_item_rec = pred_ratings.map(lambda x: (x[0], top_k(x[1], k, 1)))
    # (user_id, [item_id])
    user_item_rec = user_item_rec.map(lambda x: (x[0],
                                                 [t for t, _ in x[1]]))
    print("user_item_rec", user_item_rec.take(1))

    # 4. compute hit number of each user
    # (user_id, user_hit)
    user_item_hit = user_item_test.join(user_item_rec) \
        .map(lambda x: (x[0], len(set(x[1][0]) & set(x[1][1]))))
    print("user_item_hit", user_item_hit.take(10))

    hit_count = user_item_hit.reduce(lambda x, y: x[1] + y[1])
    test_count = test_RDD.count()
    rec_count = user_item_rec.count() * k

    # 5. compute precison recall
    precision = hit_count / (1.0 * rec_count)
    recall = hit_count / (1.0 * test_count)

    # 6. compute coverage
    # (item_id)
    all_rec_items = user_item_rec.flatMap(lambda x: x[1]).distinct()
    items_total_count = training_RDD.map(lambda x: x[1]).distinct().count()
    coverage = all_rec_items.count() / (1.0 * items_total_count)

    # 7. compute popularity
    item_popular = training_RDD.map(lambda x: (x[1], 1)) \
        .reduceByKey(add)
    popular_sum = all_rec_items.map(lambda x: (x, 1)) \
        .join(item_popular) \
        .map(lambda x: math.log(1 + x[1][1])).reduce(add)

    popularity = popular_sum / (1.0 * rec_count)

    print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
           (precision, recall, coverage, popularity))

    return (precision, recall, coverage, popularity)
