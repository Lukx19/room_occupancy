# import numpy
import keras
import pandas
from keras.layers import Dense, Lambda
from keras.models import Sequential


#from keras.wrappers.scikit_learn import KerasRegressor

#from sklearn.model_selection import KFold, cross_val_score


#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler


def loadData(filename):
    table = pandas.read_csv(filename, sep=',')
    dataset = table.values
    input = dataset[:, 1:6]
    output = dataset[:, 6]
    return (input, output)


def oneLayerModel():
    model = Sequential()
    model.add(Dense(5, input_dim=5, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal'))
    #model.add(Lambda(lambda x: round(x)))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

    # seed = 19
X, Y = loadData("data/datatraining.txt")
print("Dataset read")
print(X[0:3, :])
print(Y[0:5])
model = oneLayerModel()
print(model.to_json())
model.fit(X, Y,
          nb_epoch=100,
          batch_size=16,
          callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=False)])
print("Test data read")
X, Y = loadData("data/datatest.txt")
score = model.evaluate(X, Y, batch_size=16)
print(score)

# evaluate model with standardized dataset
# estimator = KerasRegressor(
#     build_fn=oneLayerModel, nb_epoch=30, batch_size=5, verbose=1)
#
# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
