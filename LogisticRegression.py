# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

class LogisticReg(object):

    def __init__(self, X_set, y_set):
        self.X_set = X_set
        self.y_set = y_set
        self.logi_reg = LogisticRegression()

        
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # Fit training data
        self.logi_reg.fit(X_train, y_train)
        # Predict labels via feature test set
        y_pred = self.logi_reg.predict(X_test)
        return y_test, y_pred

    
    def roc_curve(self, X_test, y_test, feature, label):
        # predict probabilities
        y_pred_prob = self.logi_reg.predict_proba(X_test)[:, 1]
        # Generate true/false positive rates and threshold values
        false_pos, true_pos, thresholds = roc_curve(y_test, y_pred_prob)
        # Plot an ROC chart
        plt.plot([0, 1], [0, 1], 'kd--')
        plt.plot(false_pos, true_pos)
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        plt.title('{} vs {} ROC Curve'.format(feature, label))
        plt.show()

    def auc_score(self, X_test, y_test, cross_val):
        # Generate Predicted Probs
        y_pred_prob = self.logi_reg.predict_proba(X_test)[:, 1]
        # Generate AUC score
        print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
        # Generate cross-validated AUC scores
        cross_val_auc = cross_val_score(self.logi_reg, self.X, self.y, cv=cross_val, scoring='roc_auc')
        # Area under the curve scores
        print("AUC scores computed using {}-fold cross-validation: {}".format(cross_val, cross_val_auc))
