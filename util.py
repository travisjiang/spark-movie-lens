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

def debug_open(func):
    def wrapper(*args, **kwargs):
        import drpyspark
        drpyspark.enable_debug_output()
        return func(*args, **kwargs)
    return wrapper

def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("item_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['util.py', 'model.py', 'engine.py', 'app.py'])

    return sc


def top_k(l, k, index):
    sorted_l = sorted(l, key=operator.itemgetter(index), reverse=True)
    if len(sorted_l) > k:
        return sorted_l[:k]
    return sorted_l


def test_model_rmse(model, test_RDD):
    '''Test model by: rmse
    Args:
        model: Recommend model, must implement predictAll(rdd)
        k: Recommend top k items for each user
        training_RDD: Traning dataset used
        test_RDD: Test dataset to evaluate model
    Returns:
        rmse of model

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


def evaluate_model(model, k, training_RDD, test_RDD):
    '''Evaluate model with: precision, recall, coverage and popularity

    Args:
        model: Recommend model, must implement predictAll(rdd)
        k: Recommend top k items for each user
        training_RDD: Traning dataset used
        test_RDD: Test dataset to evaluate model
    Returns:
        (precison, recall, coverage, popularity)

    '''

    import drpyspark
    drpyspark.enable_debug_output()

    # 1. transfer test rdd
    # (user_id, [(item_id, rating)])
    user_item_test = test_RDD.map(lambda x: (x[0], (x[1], x[2]))) \
        .groupByKey().mapValues(list) \
        .map(lambda x: (x[0], top_k(x[1], k, 1)))
    print("user_item_test1 ", user_item_test.take(1))
    # (user_id, [item_id])
    user_item_test = user_item_test.map(lambda x: (x[0],
                                                   [t for t, _ in x[1]]))
    print("user_item_test ", user_item_test.take(1))

    # 2. predict ratings
    # (user_id, [(item_id, rating)])
    pred_ratings = model.predictAll(test_RDD.map(lambda x: (x[0], x[1]))) \
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
    # (user_hit)
    user_item_hit = user_item_test.join(user_item_rec) \
        .map(lambda x: len(set(x[1][0]) & set(x[1][1])))
    print("user_item_hit", user_item_hit.take(10))

    hit_count = user_item_hit.reduce(lambda x, y: x + y)
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

def split_by_time(rdd, ratio):
    """split data for each user by ratio in timeline
    Args:
        rdd: Dataset with format(user_id, item_id, rating, time)
        ratio: Split ratio, eg.[3, 7]
    Returns:
        tuple of split rdds

    """
    #(user_id, [(item_id, rating, time)])
    sort_by_time = rdd.map(lambda x:(x[0], (x[1], x[2], x[3])))\
            .groupByKey().mapValues(list)\
            .map(lambda x:(x[0],sorted(x[1], key=operator.itemgetter(2), reverse=False)))

    def _split_f(x, ratio):
        s = sum(ratio)
        b = 0
        l = []
        for r in ratio:
            e = int(float(r)/s*len(x))
            l.append(x[b:b+e])
            b = b+e
        return l

    #(user_id, [[(item_id, rating, time)]...[]])
    split_by_ratio = sort_by_time.map(lambda x: (x[0], _split_f(x[1], ratio)))
    print("split by ratio: ", split_by_ratio.take(1))

    split_list = []
    for i, _ in enumerate(ratio):
        split_list.append(split_by_ratio.map(lambda x:(x[0], x[1][i]))\
                .flatMap(lambda x:[(x[0], y[0], y[1], y[2]) for y in x[1]]))
        print("i ", i)
        print("len ", split_list[i].count())

    return tuple(split_list)









