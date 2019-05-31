"""
model that combines syntactic features and word embeddings
"""
from SRL_utils import*
#data:
training = "data_splits/training.json"
training_df = pd.read_json(training, lines=True)

development = "data_splits/development.json"
development_df = pd.read_json(development, lines=True)

test = "data_splits/test.json"
test_df = pd.read_json(development, lines = True)

#extraction of the features for the training data
train_gold = [] #gold label
training_instances = [] #features
for file in training_df:
    gold, features = open_xml(file, version = "combined")
    train_gold.extend(gold)
    training_instances.extend(features)

#extraction of the features for the test data                
test_gold = list()
test_instances = list()
for file in test_df:
    gold, features = open_xml(file, version = "combined")
    test_gold.extend(gold)
    test_instances.extend(features)

#training the svm and running the trained svm on the test data
run_svr(training_instances, test_instances, train_gold, test_gold)
