import deepchem as dc


featurizer = dc.feat.OneHotFeaturizer()
smiles = ['CCC']
encodings = featurizer.featurize(smiles)
featurizer.untransform(encodings[0])