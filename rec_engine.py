import os, math
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext, SparkConf

from operator import add

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(ID_and_ratings_tuple):
    """Given a tuple (itemID, ratings_iterable)
    returns (itemID, (ratings_count, ratings_avg))
    """
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)


class RecommendationEngine:
    """A item recommendation engine
    """

    def __count_and_average_ratings(self):
        """Updates the items ratings counts from
        the current data self.ratings_RDD
        """
        logger.info("Counting item ratings...")
        item_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        item_ID_with_avg_ratings_RDD = item_ID_with_ratings_RDD.map(get_counts_and_averages)
        self.items_rating_counts_RDD = item_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    def __predict_ratings(self, user_and_item_RDD):
        """Gets predictions for a given (userID, itemID) formatted RDD
        Returns: an RDD with format (itemTitle, itemRating, numRatings)
        """
        if not self.items_rating_counts_RDD:
            # Pre-calculate items ratings counts
            self.__count_and_average_ratings()

        predicted_RDD = self.model.predictAll(user_and_item_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.items_titles_RDD).join(self.items_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        """Add additional item ratings in the format (user_id, item_id, rating)
        """
        # Convert ratings to an RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # Add new ratings to the existing ones
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        # Re-compute item ratings count
        self.__count_and_average_ratings()
        # Re-train the ALS model with the new ratings
        self._train_model()

        return ratings

    def get_ratings_for_item_ids(self, user_id, item_ids):
        """Given a user_id and a list of item_ids, predict ratings for them
        """
        requested_items_RDD = self.sc.parallelize(item_ids).map(lambda x: (user_id, x))
        # Get predicted ratings
        ratings = self.__predict_ratings(requested_items_RDD).collect()

        return ratings

    def _train_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Training the ALS model...")
        self.model = ALS.train(self.training_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model built!")

    def _test_model(self):

        test_for_predict_RDD = self.test_RDD.map(lambda x: (x[0], x[1]))

        predictions = self.model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = self.test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

        print 'For testing data the RMSE is %s' % (error)


    def evaluate(self, k):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        training_RDD = self.training_RDD
        test_RDD = self.test_RDD

        # (user_id, [item_id])
        user_item_test = test_RDD.map(lambda x: (x[0], x[1]))\
            .groupByKey().mapValues(list)
        print("user_test ", user_item_test.take(1))

        # (user_id, [(item_id, rating)])
        pred_ratings = self.model.predictAll(test_RDD.map(lambda x: (x[0], x[1])))\
            .groupByKey().mapValues(list)
        user_item_rec = pred_ratings.map(lambda x:(x[0],\
            sorted(x[1], key=operator.itemgetter(1), reverse=True)[:k]))
        print("user_rec1 ", user_item_rec.take(1))

        # (user_id, [item_id])
        user_item_rec = user_item_rec.map(lambda x:(x[0],\
            [t for t,_ in x[1]]))
        print("user_rec2 ", user_item_rec.take(1))

        # (user_id, user_hit)
        users_hit = user_item_test.join(user_item_rec)\
            .map(lambda x: (x[0], set(x[1][0]) & set(x[1][1])))

        hit_count = users_hit.reduceByKey(add)
        test_count = test_RDD.count()
        rec_count = user_item_rec.count()*k

        precision = hit_count / (1.0 * rec_count)
        recall = hit_count / (1.0 * test_count)

        # (item_id)
        all_rec_items = user_item_rec.flatMap(lambda x: x[1]).distinct()
        items_total_count = training_RDD.map(lambda x:x[1]).distinct().count()
        coverage = all_rec_items.count()/(1.0 * items_total_count)

        item_popular = training_RDD.map(lambda x: (x[1], 1))\
            .reduceByKey(add)
        popular_sum = all_rec_items.map(lambda x: (x, 1))\
            .join(item_popular)\
            .map(lambda x: math.log(1+x[1][1])).reduce(add)

        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity))


    def recommend_for_user(self, user_id, items_count):
        """
        Recommends up to items_count top unrated items to user_id
        Returns: List of tuple with format(id, title, rating)
        """

        # get items=>(user_id, item_id)
        unrated_items = self.ratings_RDD.filter(lambda x: x[0] != user_id)\
                .map(lambda x: (user_id, x[1])).distinct()

        # top ratings=>(item_id, ratings)
        top_items= self.model.predictAll(unrated_items)\
                .map(lambda x: (x[1], x[2]))

        # recommend items=>(item_id, item_title, ratings)
        recommend_items = top_items.join(self.items_titles_RDD)\
                .map(lambda x: (x[0], x[1][1], x[1][0]))\
                .takeOrdered(items_count, lambda x:-x[2])

        return recommend_items

    def _load_data(self, sc, dataset_path):
        logger.info("Starting up the Recommendation Engine: ")

        self.sc = sc

        # Load ratings data for later use
        # format: userid::itemid::rates::time
        # example: 1::122::5::838985046
        logger.info("Loading Ratings data...")
        ratings_file_path = os.path.join(dataset_path, 'ratings.dat')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        self.ratings_RDD = ratings_raw_RDD.map(lambda line: line.split("::"))\
                .map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

        # Load items data for later use
        # format: itemid::name::class
        # example: 1::Toy Story(1995)::Adventure|Animation|Children|Comedy|Fantasy
        logger.info("Loading items data...")
        items_file_path = os.path.join(dataset_path, 'movies.dat')
        items_raw_RDD = self.sc.textFile(items_file_path)
        self.items_RDD = items_raw_RDD.map(lambda line: line.split("::"))\
                .map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.items_titles_RDD = self.items_RDD.map(lambda x: (int(x[0]),x[1])).cache()

        self.training_RDD, self.test_RDD = self.ratings_RDD.randomSplit([7, 3], seed=0L)
        #self.training_RDD = self.ratings_RDD

    def __init__(self, sc, dataset_path, model_path):
        """Init the recommendation engine given a Spark context and a dataset path
        """
        # Load data
        self._load_data(sc, dataset_path)

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

        self._test_model()


def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("item_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])

    return sc


# this is for test
if __name__ == "__main__":

    sc = init_spark_context()
    dataset_path = os.path.join('datasets', 'ml-10M100K')

    model_path = os.path.join("./models", 'movie_lens_als')

    engine = RecommendationEngine(sc, dataset_path, model_path)

    user_id, top_k = 10, 10

    items = engine.recommend_for_user(user_id, top_k)

    for m in items:
        print("recommend for %d, id: %d, item: %s, ratings: %f" % \
            (user_id, m[0], m[1], m[2]))

    engine.evaluate(10)






