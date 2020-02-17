from tensorflow import keras
# Make sure the tf_trees directory is in the search path.
from tf_trees import TEL

# The documentation of TEL can be accessed as follows
print(TEL.__doc__)

# We will fit TEL on the Boston Housing regression dataset.
# First, load the dataset.
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Define the tree layer; here we choose 10 trees, each of depth 3.
# Note output_logits_dim is the dimension of the tree output.
# output_logits_dim = 1 in this case, but should be equal to the
# number of classes if used as an output layer in a classification task.
tree_layer = TEL(output_logits_dim=1, trees_num=10, depth=3)

# Construct a sequential model with batch normalization and TEL.
model = keras.Sequential()
model.add(keras.layers.BatchNormalization())
model.add(tree_layer)

# Fit a model with mse loss.
model.compile(loss='mse',  optimizer='adam', metrics=['mse'])
result = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
