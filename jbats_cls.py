#!/usr/bin/env python3

import vecto.embeddings
import vecto.benchmarks.analogy.analogy
import pickle

path_model = "./embeddings-cl-poincare/"
model = vecto.embeddings.load_from_dir(path_model)
options = {}
options["path_dataset"] = "./JBATS_1.0/"
ana = vecto.benchmarks.analogy.analogy.Analogy()
d = ana.run(model, options["path_dataset"])
with open('./result-cl-poincare.pickle', 'wb') as pf:
    pickle.dump(d, pf, protocol=4)

path_model2 = "./embeddings-cl-jawiki/"
model2 = vecto.embeddings.load_from_dir(path_model2)
ana2 = vecto.benchmarks.analogy.analogy.Analogy()
d2 = ana2.run(model2, options["path_dataset"])
with open('./result-cl-jawiki.pickle', 'wb') as pf:
    pickle.dump(d2, pf, protocol=4)
