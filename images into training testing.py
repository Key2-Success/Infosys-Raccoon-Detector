import os
from sklearn.cross_validation import train_test_split

# change working directory
os.chdir("C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/pumpkin")
os.getcwd()

# split data
X = y = os.listdir("original")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# move files
for each in X_train:
    os.rename("original/" + each, "train/" + each)
    
for each in X_test:
    os.rename("original/" + each, "test/" + each)