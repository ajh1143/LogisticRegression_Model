#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

class LogisticReg(object):


    def __init__(self, X_set, y_set):
        self.X_set = X_set
        self.y_set = y_set
        
        
    def attributes(self):
      print(self.__dict__)


    def reports(self, y_test, y_pred):
        # Generate Confusion Matrix and Classification Report
        conf_matx = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print("Confusion Matrix:\n{}\n").format(conf_matx)
        print("Classification Report:\n{}\n").format(class_report)


    def LogRegressor(self, X, y):
        # Train/Test Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
        # Instantiate LogisticRegression
        logi_reg = LogisticRegression()
        # Fit training data
        logi_reg.fit(X_train, y_train)
        # Predict labels via feature test set
        y_pred = logi_reg.predict(X_test)
        return y_test, y_pred
