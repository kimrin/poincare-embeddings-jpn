#!/usr/bin/env python3
import datetime
import os
import uuid
import logging
import progressbar
import sklearn
import numpy as np
import pickle
import vecto.embeddings
import vecto.benchmarks.analogy.analogy
from vecto.data import Dataset
from vecto.benchmarks.base import Benchmark
from vecto.benchmarks.analogy.io import get_pairs
from vecto.benchmarks.analogy.solvers import LinearOffset, LRCos, PairDistance
from vecto.benchmarks.analogy.solvers import ThreeCosAvg, ThreeCosMul, ThreeCosMul2
from vecto.benchmarks.analogy.solvers import SimilarToAny, SimilarToB


logger = logging.getLogger(__name__)


def select_method_poincare(key):
    if key == "3CosAvg":
        method = ThreeCosAvg
    elif key == "SimilarToAny":
        method = SimilarToAny
    elif key == "SimilarToB":
        method = SimilarToB
    elif key == "3CosMul":
        method = ThreeCosMul
    elif key == "3CosMul2":
        method = ThreeCosMul2
    elif key == "3CosAdd":
        method = LinearOffset
    elif key == "PairDistance":
        method = PairDistance
    elif key == "LRCos" or key == "SVMCos":
        method = LRCos
    elif key == '3CosAddPoincare':
        method = LinearOffsetPoincare
    else:
        raise RuntimeError("method name not recognized")
    return method


class LinearOffsetPoincare(vecto.benchmarks.analogy.solvers.PairWise):
    def la(self, vec):
        if np.dot(vec, vec) == 1.0:
            return 2.0
        return 2.0 / (1.0 - np.dot(vec, vec))

    def d_poincare_ball(self, vec_x, vec_y):
        lambda_x = self.la(vec_x)
        lambda_y = self.la(vec_y)

        d = np.arccosh(1.0 + lambda_x * lambda_y * (np.dot(vec_x - vec_y, vec_x - vec_y) * 0.5))

        return d

    def compute_scores(self, vec_a, vec_a_prime, vec_b):
        v_0 = np.zeros_like(vec_a)
        vec_ab_p = self.d_poincare_ball(v_0, vec_a + vec_b) * (vec_a + vec_b)
        vec_a_prime_p = self.d_poincare_ball(v_0, vec_a_prime) * vec_a_prime
        vec_b_prime_predicted_p = self.d_poincare_ball(vec_ab_p, vec_a_prime_p) * (vec_a_prime_p - vec_ab_p)
        vec_b_prime_predicted = self.normed(vec_b_prime_predicted_p)

        scores = self.get_most_similar_fast(vec_b_prime_predicted_p)

        return scores, vec_b_prime_predicted_p


class AnalogyPoincare(vecto.benchmarks.analogy.analogy.Analogy):
    def run(self, embs, path_dataset):  # group_subcategory
        self.embs = embs
        self.solver = select_method_poincare(
            self.method)(self.embs, exclude=self.exclude)

        if self.normalize:
            self.embs.normalize()
        self.embs.cache_normalized_copy()

        results = []
        dataset = Dataset(path_dataset)
        for filename in dataset.file_iterator():
            logger.info("processing " + filename)
            pairs = get_pairs(filename)
            name_category = os.path.basename(os.path.dirname(filename))
            name_subcategory = os.path.basename(filename)
            experiment_setup = dict()
            experiment_setup["dataset"] = dataset.metadata
            experiment_setup["embeddings"] = self.embs.metadata
            experiment_setup["category"] = name_category
            experiment_setup["subcategory"] = name_subcategory
            experiment_setup["task"] = "word_analogy"
            experiment_setup["default_measurement"] = "accuracy"
            experiment_setup["method"] = self.method
            experiment_setup["uuid"] = str(uuid.uuid4())
            if not self.exclude:
                experiment_setup["method"] += "_honest"
            experiment_setup["timestamp"] = datetime.datetime.now().isoformat()
            result_for_category = self.run_category(pairs)
            result_for_category["experiment_setup"] = experiment_setup
            results.append(result_for_category)
        # if group_subcategory:
            # results.extend(self.group_subcategory_results(results))
        return results


path_model = "./embeddings-cl-poincare/"
model = vecto.embeddings.load_from_dir(path_model)
options = {}
options["path_dataset"] = "./JBATS_1.0/"
ana = AnalogyPoincare(method='3CosAddPoincare')
d = ana.run(model, options["path_dataset"])
with open('./result-cl-poincare-new-distance.pickle', 'wb') as pf:
    pickle.dump(d, pf, protocol=4)

path_model2 = "./embeddings-cl-jawiki/"
model2 = vecto.embeddings.load_from_dir(path_model2)
ana2 = vecto.benchmarks.analogy.analogy.Analogy()
d2 = ana2.run(model2, options["path_dataset"])
with open('./result-cl-jawiki.pickle', 'wb') as pf:
    pickle.dump(d2, pf, protocol=4)
