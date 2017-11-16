
import math
import util
import logging

from itertools import combinations
import operator

import numpy as np
from pyspark.rdd import RDD
from pyspark.mllib.recommendation import Rating

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class _BaseModel:
    """Abstract object representing a recommend model

    Every recommend model must implement:
    * fit(self)
    * predict(self, user, product)
    * predictAll(self, user_product)
    * recommendProducts(self, user, num)
    * recommendProductsForUsers(self, num)

    Any other recommend model derives from this class
    """

    def __init__(self, sc, training_rdd, model_path, retrain=True):
        '''init model data
        Args:
            sc: SparkContext
            training_rdd: RDD with format (user, product, rating, time)
        '''

        self.sc = sc
        self.training_rdd = training_rdd
        self.model_path = model_path
        self.retrain = retrain

    def fit(self):
        """
        Training model based on self.training_rdd
        """
        raise NotImplementedError("Abstract method")

    def predict(self, user, product):
        """
        Predicts rating for the given user and product.
        """
        raise NotImplementedError("Abstract method")

    def predictAll(self, user_product):
        """
        Returns a list of predicted ratings for input user and product
        pairs.
        """
        assert isinstance(
            user_product, RDD), "user_product should be RDD of (user, product)"
        first = user_product.first()
        assert len(first) == 2, "user_product should be RDD of (user, product)"
        #(user_id, item_id)
        user_product = user_product.map(lambda u_p: (int(u_p[0]), int(u_p[1])))

    def recommendProducts(self, user, num):
        """
        Recommends the top "num" number of products for a given user and
        returns a list of Rating objects sorted by the predicted rating in
        descending order.
        """
        raise NotImplementedError("Abstract method")

    def recommendProductsForUsers(self, num):
        """
        Recommends the top "num" number of products for all users. The
        number of recommendations returned per user may be less than "num".
        """
        raise NotImplementedError("Abstract method")


class RandomModel(_BaseModel):
    """Random model

    """

    def __init__(self, sc, training_rdd, model_path, retrain=True):
        super(
            RandomModel,
            self).__init__(
            sc,
            training_rdd,
            model_path,
            retrain)

    def fit(self):
        self.max_rating = self.training_rdd.max()
        self.min_rating = self.training_rdd.min()

        self.item_set = self.sc.broadcast(
            self.training_rdd.map(
                lambda x: x[1]).collect())

    def predict(self, user, product):
        """
        Predicts rating for the given user and product.

        Args:
            user: user id, int
            product: product id, int
        Returns:
            rating: predicted rating, float
        """
        return np.random.uniform(self.min_rating, self.max_rating)

    def predictAll(self, user_product):
        """
        Returns a list of predicted ratings for input user and product
        pairs.

        Args:
            user_product: rdd with format (user_id, item_id, ...)
        Returns:
            rdd with format (user_id, item_id, rating)
        """
        user_product = super(RandomModel, self).predictAll(user_product)

        pred_ratings = user_product.map(
            lambda x: Rating(
                x[0],
                x[1],
                np.random.uniform(
                    self.min_rating,
                    self.max_rating)))
        return pred_ratings

    def recommendProducts(self, user, num):
        """
        Recommends the top "num" number of products for a given user and
        returns a list of Rating objects sorted by the predicted rating in
        descending order.

        """
        #(user_id, [item_id]
        rated_list = self.training_rdd.filter(
            lambda x: x[0] != user).groupByKey().mapValues(list)

        rec_list = rated_list.map(
            lambda x: (
                x[0],
                self._random_sample(
                    x[1],
                    num))) .flatMap(
            lambda x: [
                Rating(
                    x[0],
                    p[0],
                    p[1]) for p in x[1]]).collect()

        #[Rating]
        return rec_list

    def recommendProductsForUsers(self, num):
        """
        Recommends the top "num" number of products for all users. The
        number of recommendations returned per user may be less than "num".

        Args:
            num: int
        Returns:
            rdd with format (user_id, [Rating(user,product,rating)])
        """

        rated_list = self.training_rdd.map(
            lambda x: (x[0], x[1])).groupByKey().mapValues(list)

        rec_list = rated_list.map(
            lambda x: (
                x[0],
                self._random_sample(
                    x[1],
                    num))) .flatMap(
            lambda x: [
                (x[0],
                 Rating(
                     x[0],
                     p[0],
                     p[1])) for p in x[1]])

        #(user_id, [Rating])
        return rec_list

    def _random_sample(self, rated_list, n):
        unrated_list = self.item_set - set(rated_list)
        unrated_list = [
            (t,
             np.random.uniform(
                 self.min_rating,
                 self.max_rating)) for t in unrated_list]

        return util.top_k(unrated_list, n, 1)


class PopularModel(_BaseModel):
    """Popular model

    """

    def __init__(self, sc, training_rdd, model_path, retrain=True):
        super(
            PopularModel,
            self).__init__(
            sc,
            training_rdd,
            model_path,
            retrain)

    def fit(self):

        self.item_popular = self.sc.broadcast(
            self.training_rdd.map(
                lambda x: (
                    x[1], 1)).reduceByKey(add).sortedBy(
                lambda x: x[1]).collect())

    def recommendProducts(self, user, num):
        """
        Recommends the top "num" number of products for a given user and
        returns a list of Rating objects sorted by the predicted rating in
        descending order.
        """
        #(user_id, [item_id]
        rated_list = self.training_rdd.filter(
            lambda x: x[0] != user).groupByKey().mapValues(list)

        rec_list = rated_list.map(
            lambda x: (
                x[0],
                self._popular_sample(
                    x[1],
                    num))) .flatMap(
            lambda x: [
                Rating(
                    x[0],
                    p[0],
                    p[1]) for p in x[1]]).collect()

        #[Rating]
        return rec_list

    def recommendProductsForUsers(self, num):
        """
        Recommends the top "num" number of products for all users. The
        number of recommendations returned per user may be less than "num".

        Args:
            num: int
        Returns:
            rdd with format (user_id, [Rating(user,product,rating)])
        """

        rated_list = self.training_rdd.map(
            lambda x: (x[0], x[1])).groupByKey().mapValues(list)

        rec_list = rated_list.map(
            lambda x: (
                x[0],
                self._popular_sample(
                    x[1],
                    num))) .flatMap(
            lambda x: [
                (x[0],
                 Rating(
                     x[0],
                     p[0],
                     p[1])) for p in x[1]])

        #(user_id, [Rating])
        return rec_list

    def _popular_sample(self, rated_list, n):
        s = set(rated_list)
        unrated_list = []
        for t, p in self.item_popular:
            if t not in s:
                unrated_list.append((t, p))
            if len(unrated_list) == n:
                break
        return unrated_list


class ItemCFModel(_BaseModel):
    """ItemBased CF model
    * train
    * update
    * predict_rating
    * predict_top_n
    """

    def __init__(self, sc, training_rdd, model_path, retrain=True):
        super(
            ItemCFModel,
            self).__init__(
            sc,
            training_rdd,
            model_path,
            retrain)

    def fit(self):
        self.item_sim_dict = self._compute_similarity(self.training_rdd)

        #(user_id, [(item_id, rating)])
        user_items = self.training_rdd.map(lambda x: (x[0], (x[1], x[2]))) \
            .groupByKey().mapValues(list)

        #(user_id, [(unrated_item, rating)])
        self.item_pred_ratings = user_items.map(lambda x: (
            x[0], self._compute_ratings(x[1], self.item_sim_dict)))

    def predict(self, user, product):
        """
        Predicts rating for the given user and product.

        Args:
            user: user id, int
            product: product id, int
        Returns:
            rating: predicted rating, float
        """
        item_rating = self.item_pred_ratings.filter(lambda x: x[0] == user) \
            .flatMap(lambda x: [(t, r) for t, r in x[1]]) \
            .filter(lambda x: x[0] == product).collect()
        return item_rating[1]

    def predictAll(self, user_product):
        """
        Returns a list of predicted ratings for input user and product
        pairs.

        Args:
            user_product: rdd with format (user_id, item_id, ...)
        Returns:
            rdd with format (user_id, item_id, rating)
        """
        #(user_id, item_id)
        user_product = super(ItemCFModel, self).predictAll(user_product)

        #((user_id, item_id), (rating, 1))
        pred_ratings = self.item_pred_ratings \
            .flatMap(lambda x: [((x[0], t), r) for t, r in x[1]]
                     .join(user_product.map(lambda x: ((x[0], x[1]), 1))))

        format_pred = pred_ratings.map(
            lambda x: Rating(
                x[0][0], x[0][1], x[1][0]))

        return format_pred

    def recommendProducts(self, user, num):
        """
        Recommends the top "num" number of products for a given user and
        returns a list of Rating objects sorted by the predicted rating in
        descending order.
        """
        #(user_id, [(item_id, rating)]
        rec_list = self.item_pred_ratings.filter(lambda x: x[0] == user) \
            .map(lambda x: (x[0], util.top_k(x[1], num, 1)))

        #[Rating]
        format_rec = rec_list.map(
            lambda x: [
                Rating(
                    x[0],
                    t,
                    r) for t,
                           r in x[1]])
        return format_rec

    def recommendProductsForUsers(self, num):
        """
        Recommends the top "num" number of products for all users. The
        number of recommendations returned per user may be less than "num".

        Args:
            num: int
        Returns:
            rdd with format (user_id, [Rating(user,product,rating)])
        """
        #(user_id, [(item_id, rating)]
        rec_list = self.item_pred_ratings.map(
            lambda x: (x[0], util.top_k(x[1], num, 1)))

        #[Rating]
        format_rec = rec_list.map(
            lambda x: [
                Rating(
                    x[0],
                    t,
                    r) for t,
                           r in x[1]])
        return format_rec

    @staticmethod
    def _compute_ratings(items_with_rating, sim_dict):
        """
        items_with_rating: [(item_id, rating)]
        sim_dict:          {item_id:[(item_id, sim)]}
        Returns:           [(unrated_item_id, rating)]
        """
        pred_ratings = {}
        sim_sums = {}
        rated_set = set([t for t, _ in items_with_rating])
        for rated_item, rating in items_with_rating:
            sim_items = sim_dict.get(rated_item, None)
            if sim_items:
                for sim_item, sim in sim_items:
                    if sim_item in rated_set:
                        continue
                    if sim_item not in pred_ratings:
                        pred_ratings[sim_item] = 0
                        sim_sums[sim_item] = 0
                    pred_ratings[sim_item] += sim * rating
                    sim_sums[sim_item] += sim
        pred_ratings = [(unrated_item, rating / sim_sums[unrated_item])
                        for unrated_item, rating in pred_ratings]

        return sorted(pred_ratings, key=operator.itemgetter(1), reverse=True)

    @staticmethod
    def _compute_similarity(rdd):
        """Compute sim using rdd with format(user_id, item_id, ratings)
        Returns dict with format{item_id: [(item_id, sim)]}
        """

        #(user_id, [(item_id, rating)])
        user_items = rdd.map(lambda x: (x[0], (x[1], x[2]))) \
            .groupByKey().mapValues(list)
        #([((item1, item2), (rating1, rating2))]
        item_pairs = user_items.flatMap(lambda x: _find_item_pairs(x[1]))

        #((item1, item2), [(rating1, rating2)])
        item_pairs = item_pairs.groupByKey().mapValues(list)

        #((item1, item2), sim)
        item_sims = item_pairs.map(lambda x: (x[0], _pearson_sim(x[1])))

        #(item1, [(item, sim)])
        item_sims = item_sims.flatMap(
            lambda x: [
                (x[0][0],
                 (x[0][1],
                  x[1])),
                (x[0][1],
                 (x[0][0],
                  x[1]))]) .distinct().groupByKey().mapValues(list)

        item_sim_dict = {}
        for t, ts in item_sims.collect():
            item_sim_dict[t] = ts
        return item_sim_dict

    @staticmethod
    def _pearson_sim(ratings):
        """pearson similarity"""
        ratings1, ratings2 = zip(*ratings)
        x, y = np.array(ratings1), np.array(ratings2)
        up = sum((x - x.mean()) * (y - y.mean()))
        down = math.sqrt(sum(x**2) * sum(y**2))
        return up / down

    @staticmethod
    def _find_item_pairs(item_with_rating):
        return [((item1[0], item2[0]), (item1[1], item2[1]))
                for item1, item2 in combinations(item_with_rating, 2)]
