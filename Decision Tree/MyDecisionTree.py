import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        # print ("The root is : ", self.root)
        # cur_node = self.root
        # print (cur_node)
        # cur_node = Tree_node()
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        # self.root = self.generate_tree(test_x,prediction)

        for i in range(len(test_x)):
            cur_node = self.root
            # prob = np.divide(np.bincount(np.array(test_x[i]).astype('int')),len(test_x[i]))

            while cur_node.is_leaf != True:
                if test_x[i][cur_node.feature] == 1.0:
                    cur_node = cur_node.right_child
                else:
                    cur_node = cur_node.left_child
            prediction[i] = cur_node.label

            # prediction[i] = cur_node
            # if prob[0] > prob[1]:
            #     print (test_x[i, cur_node.feature])
            #     cur_node = cur_node.left_child
            # if prob[0] < prob[1]:
            #     cur_node = cur_node.right_child
            # if cur_node.left_child == None:
            #     if cur_node.right_child == None:
            #         prediction[i] = cur_node.label

            # prob1 = np.divide(np.bincount(test_x[i]),len(test_x[i]))
            # print (prob1)
            # pass
            # print (prediction[i])
            # Tree_node.__init__(self.root)
            # prediction[i] =
            # self.root = test_x[i]
            # print("The testing is : ",len(test_x[i]))
            # traverse the decision-tree based on the features of the current sample
            # print (test_x)
            # pass # placeholder
        # prediction = self.root
        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()
        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
        # print (cur_node.is_leaf)
        # determine if the current node is a leaf node
        # if cur_node.left_child == None and cur_node.right_child == None:
        if cur_node.left_child == None:
            if cur_node.right_child == None:
                cur_node.is_leaf = True

        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            cur_node.label = label
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature
        data0 = data[data[:, cur_node.feature] == 0.0]
        label0 = label[data[:, cur_node.feature] == 0.0]
        data1 = data[data[:, cur_node.feature] == 1.0]
        label1 = label[data[:, cur_node.feature] == 1.0]
        cur_node.left_child = self.generate_tree(data0, label0)
        cur_node.right_child = self.generate_tree(data1, label1)
        # self.generate_tree(data1, label1)
        # data = data[data[:, cur_node.feature] == 0.0]
        # self.generate_tree(data[], label[data[data[:, cur_node.feature] == 0.0]])
        # print (label[data[data[:, cur_node.feature] == 0.0]])
        # print (data[data[:, cur_node.feature] == 1.0])
        # print ("The data is : ",len(data))
        # self.generate_tree(data[data[:, cur_node.feature] == 0.0], label[data[data[:, cur_node.feature] == 0.0]])
        # self.generate_tree(data[data[:, cur_node.feature] == 1.0], label[data[data[:, cur_node.feature] == 1.0]])
        # self.generate_tree([data[:, cur_node.feature] == 1.0], label[data[:, cur_node.feature] == 1.0])
        # self.generate_tree([data[:, cur_node.feature] == 0.0], label[data[:, cur_node.feature] == 0.0])
        # self.generate_tree([data[:, cur_node.feature] == 1.0], label[data[:, cur_node.feature] == 1.0])
        # split the data based on the selected feature and start the next level of recursion
        # node_entropy = self.compute_split_entropy(label[data[:, cur_node.feature] == 0.0], label[data[:, cur_node.feature] == 1.0])
        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        arr = np.ones(len(data[0]))
        for i in range(len(data[0])):
            arr[i] = self.compute_split_entropy(label[data[:, i] == 0.0], label[data[:, i] == 1.0])

        best_feat = np.argmin(arr)
            # compute the entropy of splitting based on the selected features
            # pass

            # select the feature with minimum entropy

        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches
        w_left = len(left_y)/(len(left_y)+len(right_y))
        w_right = len(right_y)/(len(left_y)+len(right_y))
        split_entropy = np.dot(w_left,self.compute_node_entropy(left_y))+np.dot(w_right,self.compute_node_entropy(right_y))

        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        prob = np.divide(np.bincount(label),len(label))
        node_entropy = -1.0*np.sum(np.multiply(prob,np.log2(prob+1e-15)))

        return node_entropy
