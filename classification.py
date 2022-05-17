from prettytable import PrettyTable
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from get_confusion_matrix import get_confusion_matrix


def classification(data_final, y):
    X = preprocessing.StandardScaler().fit(data_final).transform(data_final.astype(float))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    table = PrettyTable()
    table.field_names = ['Algorithm', 'Accuracy', 'Recall', 'Precision', 'F-Score']

    # Decision Tree
    # Initialise Decision Tree classifier and predict
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drugTree.fit(X_train, y_train)
    y_pred = drugTree.predict(X_test)
    # Call confusion matrix and accuracy
    algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('Decision', y_pred, y_test)

    # Add values to table
    table.add_row([algorithm, round(accuracy, 5), round(recall, 5),
                   round(precision, 5), round(f_score, 5)])

    # SVM
    clf = svm.SVC(kernel='rbf')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    # Call confusion matrix and accuracy
    algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('SVM', y_pred, y_test)

    # Add values to table
    table.add_row([algorithm, round(accuracy, 5), round(recall, 5),
                   round(precision, 5), round(f_score, 5)])

    # Logic Regression
    # Train and predict logistic regression model
    LR = LogisticRegression(C=0.01, solver='liblinear')
    y_pred = LR.fit(X_train, y_train).predict(X_test)
    # Call confusion matrix and accuracy
    algorithm, accuracy, recall, precision, f_score = get_confusion_matrix('LR', y_pred, y_test)

    # Add values to table
    table.add_row([algorithm, round(accuracy, 5), round(recall, 5),
                   round(precision, 5), round(f_score, 5)])

    return table
