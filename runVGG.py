from VGGHelper import split_data_set_gray_scale, get_VGG_model_features, train_simple_nn
import numpy as np

X_train, X_test, y_train, y_test, X_val,y_val = split_data_set_gray_scale('./dataset/data','./dataset/labels', 0.95, [501, 501])
#Train weights
features,dataset,VGG_model = get_VGG_model_features(X_train,y_train,[501,501])

#Prepare data for training
dataset = dataset.sample(frac=0.5, random_state=1)
#Redefine X and Y for Random Forest
X_for_training = dataset.drop(labels = ['Label'], axis=1)
X_for_training = X_for_training.values  #Convert to array
Y_for_training = dataset['Label']
Y_for_training = Y_for_training.values  #Convert to array
mapping = {0: 0, 128: 1, 255: 2}
# Vectorized mapping
Y_for_training = np.vectorize(mapping.get)(Y_for_training)

#Train NN 
train_simple_nn(300, X_for_training, Y_for_training, 'NN_Model_VGG_Features_V05')