# initialization procedure: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

class Perceptron:

    def __init__(self, label, input_dim, output_dim, step_size, num_classes=1):
        self.classifier_label = label
        self.input_len = input_dim
        self.output_len = output_dim
        self.sigmoid = lambda z : 1/(1+exp(-z))
        self.num_classes = num_classes
        self.multi_class = num_classes > 1
        self.vexp = np.vectorize(exp)




    def fit(self, X, y, weights=None, step_size=0.01, batch_size=10):
        """
        initializes training data and hyperparameters
        """
        # init weights and step_size
        assert X.shape[0] == y.shape[0]
        self.train_nobs = X.shape[0]
        if weights not None:
            self.W = weights
        else:
            self.W = np.random.randn(self.input_len, self.num_classes)*sqrt(2/(1+self.input_len))
        self.step_size = step_size
        self.batch_size = batch_size
        self. shuffler = np.random.randn(self.train_nobs)
        self.X = X[self.shuffler]
        self.y = y[self.shuffler]



    def predict(self, X):
        """
        takes a test set and returns predictions
        """
        if self.multi_class:
            return self.softmax(X.dot(self.W))
        else:
            return self.sigmoid(X.dot(self.W))

    def train(self, num_epochs=1, cost_funct='cross_ent'):
        """
        implements backpropagation algorithm
        """
        batches = [(n,n+self.batch_size) for n in range(self.input_len)]
        for a,b in batches:
            XW = X.dot(self.W)
            preds = self.predict(self.X[a:b])
            #cost = self.cost(self.y[a:b], preds, funct=cost_funct)
            cost_deriv = preds - self.y
            self.W = self.W - self.step_size *
            if self.multi_class:
                act_deriv = self.soft_deriv(XW)
            else:
                act_deriv = self.sigmoid(XW)(1-self.sigmoid(XW))
            update = X.dot(act_deriv).dot(cost_deriv)
            self.W = self.W - self.step_size * update


    def softmax(self, vector):
        denom = np.sum(self.vexp(vector))
        return np.array(self.vexp(exp))/denom

    def cost(self, y, yhat, funct='cross_ent'):
        if funct == 'cross_ent':
            return np.sum(np.vectorize(log)(yhat) * y)

    def soft_deriv(self, inputs):
        size = max(*inputs.shape)
        deriv = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                if i==j:
                    deriv[i,j] = self.sigmoid(inputs[j])(1-self.sigmoid(inputs[i]))
                else:
                    deriv[i, j] = -self.sigmoid(inputs[j]) * self.sigmoid(inputs[i])
        return deriv

#class MultiClassPerceptron(Perceptron):

#    def __init__(self):
#        pass
