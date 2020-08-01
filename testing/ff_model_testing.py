from classifier.nn_ff import FeedForward


model = FeedForward(28, 9, 3)
model.fit()
model.predict()



