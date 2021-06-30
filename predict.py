import numpy as np
import csv
import sys
import pickle
from validate import validate

train_X_file_path = "./train_X_rf.csv"
train_Y_file_path = "./train_Y_rf.csv"
numSamples = [5, 6, 7, 8, 9, 10, 11, 12, 13]

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

    def setAttrThresh(self, attr, thresh):
        self.feature_index = attr 
        self.threshold = thresh

    def setLeftChild(self, child):
        self.left = child 
        
    def setRightChild(self, child):
        self.right = child

##################################################################################################
#Functions to find best split using chosen features

def calculate_entropy(Y):
    Ys = np.unique(Y)
    entropy = 0 
    N = len(Y)
    for y in Ys:
        num = np.sum(np.array(Y) == y)
        ratio = num/N
        entropy += (-ratio)*np.log2(ratio)
    return entropy

def calculate_information_gain(Y_subsets):
    flatY = []
    for y in Y_subsets:
        flatY.extend(y)
    root = calculate_entropy(flatY)
    numElts = len(flatY)
    for y in Y_subsets:
        root -= (len(y)/numElts)*calculate_entropy(y)
    return root

def calculate_split_entropy(Y_subsets):
    Y = np.concatenate(Y_subsets)
    numElts = len(Y)
    sum = 0
    for y in Y_subsets:
        r = (len(y)/numElts)
        sum -= r*np.log2(r)
    return sum 

def calculate_gain_ratio(Y_subsets):
    return calculate_information_gain(Y_subsets)/calculate_split_entropy(Y_subsets)

def calculate_gini_index(Y_subsets):
    gini_index = 0
    total_instances = sum(len(Y) for Y in Y_subsets)
    classes = sorted(set([j for i in Y_subsets for j in i]))

    for Y in Y_subsets:
        m = len(Y)
        if m == 0:
            continue
        count = [Y.tolist().count(c) for c in classes]
        gini = 1.0 - sum((n / m) ** 2 for n in count)
        gini_index += (m / total_instances)*gini
    
    return gini_index

def split_data_set(data_X, data_Y, feature_index, threshold):
    X = np.array(data_X)
    Y = np.array(data_Y)
    left = X[:,feature_index] < threshold
    right = X[:,feature_index] >= threshold
    return X[left], Y[left], X[right], Y[right]

def get_best_split(X, Y, features):
    npX = np.array(X)
    best_feature = 0
    best_threshold = 0
    minGini = np.inf
    for feature in features:
        thresholds = np.unique(npX[:,feature])
        for threshold in thresholds:
            _, leftY, _, rightY = split_data_set(X, Y, feature, threshold)
            gini = calculate_gini_index([leftY, rightY])
            if gini < minGini:
                best_feature = feature 
                best_threshold = threshold
                minGini = gini
                
    return best_feature, best_threshold

##################################################################################################

def preorder(node):
    if node == None:
        return
    print(f"X{node.feature_index} {node.threshold}")
    preorder(node.left)
    preorder(node.right)

def predictClass(root, X):
    feature = root.feature_index 
    threshold = root.threshold 
    if X[feature] < threshold:
        if root.left != None:
            return predictClass(root.left, X)
        else: 
            return root.predicted_class
    else:
        if root.right != None:
            return predictClass(root.right, X)
        else: 
            return root.predicted_class

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_rf.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    Y = []
    for x in test_X:
        cur = []
        for root in model:
            cur.append(predictClass(root, x))
        Y.append(max(cur, key=cur.count))
    return np.array(Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def get_bootstrap_samples(train_XY, num_bootstrap_samples):
    import numpy as np
    bootstrapSamples = []
    train_XY = np.array(train_XY)
    for i in range(num_bootstrap_samples):
        bootstrapSample = []
        for j in range(len(train_XY)):
            bootstrapSample.append(train_XY[np.random.randint(len(train_XY)),:].tolist())
        bootstrapSamples.append(bootstrapSample)
    return bootstrapSamples

def get_split_in_random_forest(X, Y, num_features):
    chosenFeatures = []
    for i in range(num_features):
        chosenFeatures.append(np.random.randint(len(X[0])))
    
    return get_best_split(X, Y, chosenFeatures)

def buildTree(X, Y, max_depth, min_size, depth, num_features):
    if max_depth <= 0 or min_size == np.inf or depth >= max_depth:
        return None
    curNode = Node(np.bincount(Y).argmax(), 0)
    best_feature, best_threshold = get_split_in_random_forest(X,Y, num_features)
    leftX, leftY, rightX, rightY = split_data_set(X,Y,best_feature,best_threshold)
    curNode.setAttrThresh(best_feature, best_threshold)
    if len(leftX) <= min_size or np.sum(np.array(leftY) != leftY[0]) == 0:
        curNode.setLeftChild(None)
    else:
        curNode.setLeftChild(buildTree(leftX, leftY, max_depth, min_size, depth+1, num_features))
    if len(rightX) <= min_size or np.sum(np.array(rightY) != rightY[0]) == 0:
        curNode.setRightChild(None)
    else:
        curNode.setRightChild(buildTree(rightX, rightY, max_depth, min_size, depth+1, num_features))
    return curNode

def get_out_of_bag_error(models, train_XY, bootstrap_samples):
    eoob = 0 
    P = 0
    for xy in train_XY:
        z = 0
        wrongPreds = 0 
        for i in range(len(bootstrap_samples)):
            if any((bootstrap_samples[i][:] == xy).all(1)):
                z += 1 
                if predictClass(models[i], xy[:-1]) != xy[-1]:
                    wrongPreds += 1 
        
        if z != 0:
            P += 1 
            eoob += (wrongPreds/z)
    return eoob/P

def trainModel():
    train_X = np.genfromtxt(train_X_file_path, delimiter=",", skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=",", skip_header=0, dtype=np.int64)
    train_XY = np.zeros((train_X.shape[0],train_X.shape[1]+1))
    train_XY[:,:-1] = train_X 
    train_XY[:,-1:] = train_Y.reshape(len(train_Y),1)
    maxDepth = np.inf
    minSize = 1
    minEOOB = np.inf 
    bestModel = None
    num_features = train_X.shape[1]
    for num_bootstrap_samples in numSamples:
        bootstrapSamples = get_bootstrap_samples(train_XY, num_bootstrap_samples)
        roots = []
        for sample in bootstrapSamples:
            sample = np.array(sample)
            X = sample[:,:-1].tolist()
            Y = np.array(sample[:,-1:].reshape(len(sample),), dtype=np.int64).tolist()
            root = buildTree(X, Y, maxDepth, minSize, 0, num_features)
            roots.append(root)
        eoob = get_out_of_bag_error(roots, train_XY, bootstrapSamples)
        print(f"num_bootstrap_samples : {num_bootstrap_samples}, curEOOB : {eoob}, minEOOB : {minEOOB}")
        if eoob < minEOOB:
            minEOOB = eoob 
            bestModel = roots
    return bestModel

def predict(test_X_file_path):
    if "-trainModel" in sys.argv:
        root = trainModel()
        pickle.dump(root, open("MODEL_FILE.sav", "wb"))
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_rf.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_rf.csv") 