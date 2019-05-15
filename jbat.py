#!/usr/bin/env python3

import vecto.embeddings
import vecto.benchmarks.analogy.analogy
import pickle

path_model = "./embeddings-jawiki/"
model = vecto.embeddings.load_from_dir(path_model)
options = {}
options["path_dataset"] = "./JBATS_1.0/"
ana = vecto.benchmarks.analogy.analogy.Analogy()
d = ana.run(model, options["path_dataset"])
with open('./result.pickle', 'wb') as pf:
    pickle.dump(d, pf, protocol=4)

