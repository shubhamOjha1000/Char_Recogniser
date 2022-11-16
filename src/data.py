import pandas as pd
import config
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


def elt_data():
     """Extract, load and transform our data assets."""

     # Extract
     train_data = pd.read_csv(config.data_path)
     label = pd.read_csv(config.label_path)

     #Load
     train_data.to_csv('/Users/shubhamojha/Desktop/Char_Recogniser/DATA/train_features.csv' , index=False, header = None)
     label.to_csv('/Users/shubhamojha/Desktop/Char_Recogniser/DATA/labels.csv' , index=False, header = None)

     #Transform
     
     label = label.rename(columns = {0 : 'labels'}) 

     train_data = train_data/ 255.0

     train_data= train_data.values.reshape(-1,28,28,1)

     label  = to_categorical(label , num_classes = 10)

     random_seed = 2
     X_train, X_val, Y_train, Y_val = train_test_split(train_data, label, test_size = 0.1, random_state=random_seed)
     return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    elt_data()




