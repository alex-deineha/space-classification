import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



filters = ['u', 'g', 'r', 'i', 'z']

categorical_columns = [f'{filter}_6' for filter in filters]
binary_columns = ['clean']
real_columns = ['rowc', 'colc', 'rowv', 'colv', 'ra', 'dec'] \
                     + [f'{filter}_{i}' for filter in filters for i in range(6)]
to_drop_columns = ['ra', 'dec', 'class']


def preprocess(data):
  data = data.copy()
  
  data = data.replace('na', np.nan)

  data[categorical_columns] = data[categorical_columns].astype(np.int64)
  data[binary_columns] = data[binary_columns].astype(np.int64)
  data[real_columns] = data[real_columns].astype(np.float64)
  
  
  #fillna
  data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].median())
  data[binary_columns] = data[binary_columns].fillna(data[binary_columns].mean())
  data[real_columns] = data[real_columns].fillna(data[real_columns].mean())
  
  #simplify fiter_6
  g_6 = dict(zip(range(9), [0,1,0,2,2,2,3,3,3]))
  r_6 = dict(zip(range(9), [0,1,2,2,2,2,2,2,3]))
  i_6 = dict(zip(range(9), [0,1,1,2,2,2,2,3,3]))
  z_6 = dict(zip(range(9), [0,0,0,1,1,1,2,3,2]))

  data['g_6'] = data['g_6'].map(g_6)
  data['r_6'] = data['r_6'].map(r_6)
  data['i_6'] = data['i_6'].map(i_6)
  data['z_6'] = data['z_6'].map(z_6)
  
  #code cat features
  for i in categorical_columns:
    data = data.join(pd.get_dummies(data[i], prefix=i))
  data = data.drop(columns=categorical_columns)
  
  #add velocity features
  data['abs_colv'] = data['colv'].abs()
  data['abs_rowv'] = data['rowv'].abs()
  data['radius'] = np.square(data['rowv']) + np.square(data['colv'])
  
  
  #add dbscan features
  from sklearn.cluster import DBSCAN
  
  labels = DBSCAN(eps=0.4, min_samples=150, n_jobs=-1).fit_predict(data[['ra', 'dec']])
  data['dbscan_cluster'] = (labels != -1).astype(np.int64)
  data = data.drop(columns=['ra', 'dec'])
    
  #drop useless features
  data = data.drop(columns=['rowc', 'colc'])
  data = data.drop(columns=[f'{filter}_{i}' for filter in ['g', 'r', 'i', 'z']for i in range(1,3)])
  data = data.drop(columns=[f'{filter}_4' for filter in filters])
  
  #scale data
  from sklearn.preprocessing import scale
  data = pd.DataFrame(scale(data.values), index=data.index, columns=data.columns)
  return data



import keras  
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l2

def create_model(f):
  
  reg = l2(0.0001)
  
  model = Sequential()
  model.add(Dense(units=32, activation='relu', input_dim=f, kernel_regularizer=reg))
  model.add(Dropout(0.2))
  model.add(Dense(units=16, activation='relu', kernel_regularizer=reg))
  model.add(Dropout(0.1))
  model.add(Dense(units=3, activation='softmax', kernel_regularizer=reg))

  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.0005),
                metrics=['accuracy'])
  
  return model





import sys


def main(path_to_train, path_to_unlabeled_data, path_to_test, path_to_predictions):
    train_data = pd.read_csv(path_to_train, index_col=0)
    test_data = pd.read_csv(path_to_test, index_col=0)

    
    y_train = pd.get_dummies(train_data['class'])
    train_data = train_data.drop('class', axis=1)
    
    data=pd.concat([train_data, test_data])

    features = preprocess(data)

    X_train, X_test = features.loc[train_data.index], features.loc[test_data.index]
    
    
    
    model = create_model(X_train.shape[1])

    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.999,
                                  patience=3, min_lr=0.0001, verbose=1)

    model.fit(X_train, y_train, 
                        epochs=150, batch_size=32, callbacks = [reduce_lr])
    
    
    y_pred = model.predict(X_test, batch_size=128)
    result = pd.DataFrame(y_pred.argmax(axis=1), index=test_data.index, columns=['prediction'])
    
    result.to_csv(path_to_predictions)
    

if __name__ == "__main__":
    path_to_train = str(sys.argv[1])
    path_to_unlabeled_data = str(sys.argv[2])
    path_to_test = str(sys.argv[3])
    path_to_predictions = str(sys.argv[4])
    
    main(path_to_train, path_to_unlabeled_data, path_to_test, path_to_predictions)
