#!/usr/bin/python
#coding=utf-8

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation as val
import numpy
import cPickle as cpkl
import matplotlib
from data_reader import DataReader
import config
import os

## Главным классфикатором является классификатор clf
## для тестирования есть classifier (см. метод train)

class SVMTrainer(object):
    def __init__(self, train_data_path, val_data_path):
        self.tpath = train_data_path
        self.vpath = val_data_path

    def read_data(self):
        if hasattr(self, "X_train")\
            and hasattr(self, "Y_train")\
            and hasattr(self, "X_test")\
            and hasattr(self, "Y_test"):
                return [(self.X_train, self.X_val), (self.Y_train, self.Y_val)]

        train_data_reader = DataReader(self.tpath)
        val_data_reader = DataReader(self.vpath)
        self.X_train, self.X_val = train_data_reader.get_objects(), val_data_reader.get_objects()
        self.Y_train, self.Y_val = train_data_reader.get_classes(), val_data_reader.get_classes()

        return [(self.X_train, self.X_val), (self.Y_train, self.Y_val)]

    ##оптимизация классификатора
    def eval_estimator(self):
        if hasattr(self, "classification_rep"):
            return self.classification_rep

        ### tuned parameters
        C = [i for i in range(1, 100)]
        intercept_scaling=[i for i in range (1, 10)]
        tol = [0.001, 0.00001, 0.00001, 0.00001]
        parameters = [{'C':self.C,
                       'class_weight':['auto', None],
                       'intercept_scaling':intercept_scaling,
                       'tol':self.tol,
                       'dual':[False, True]}]
        cv=5

        ##evaluated scores
        scores = ['accuracy',
                  'average_precision',
                  'precision',
                  'recall',
                  'f1',
                  'roc_auc']

        for score in scores:
            print("Tuning hyper-parameters for {0}\n".format(score))
            self.clf=GridSearchCV(svm.LinearSVC(), parameters, cv=cv, scoring=score)
            self.clf.fit(self.X_train, self.Y_train)

            print("Best parameters (with train set):\n")
            print(self.clf.best_estimator_)
            print("Grid scores (with train set):\n")
            for params, mean_score, scores in self.clf.grid_scores_:
                print("%0.5f (+/-%0.05f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print("Detailed classification report:\n")
            print("The model is trained on the full train set.\n")
            print("The scores are computed on the full validation set.\n")
            print("\n")
            real_classes, pred_classes = self.Y_val, self.clf.predict(self.X_val)
            self.classification_rep = classification_report(real_classes, pred_classes)
            return self.classification_rep

    #Этот метод будет не нужен в последствии
    #Так как после подгонки параметров классификатор будет натренирован
    def train(self):
        if hasattr(self, 'classifier'):
            return self.classifier
        ## Параметры получены ранее после оптимизации
        self.classifier = svm.LinearSVC(C=7,
                                   class_weight=None,
                                   dual=True,
                                   fit_intercept=True,
                                   intercept_scaling=1,
                                   loss='l2',
                                   multi_class='ovr',
                                   penalty='l2',
                                   random_state=None,
                                   tol=0.0001,
                                   verbose=0)
        self.classifier.fit(self.X_train,self.Y_train)
        return self.classifier

    def dump_classifier(self):
         ## после оптимизации параметров поменять на
        ##if (hasattr(self,clf)):
        if (hasattr(self,'classifier')):
            if not os.path.exists(config.SVM_CLF_DIR):
	            os.makedirs(config.SVM_CLF_DIR)

            output_clf = open(config.SVM_CLF_DIR+"/trained_classifier.pkl", "wb")
            cpkl.dump(self.classifier, output_clf)
            output_clf.close()
        else:
            print("dump_classifier : train classifier first\n")

if __name__ == '__main__':
    svm_obj = SVMTrainer(config.TRAIN_DATA_PATH, config.VAL_DATA_PATH)
    svm_obj.read_data()
    svm_obj.train()
    svm_obj.dump_classifier()

#print("error percentage (underfitting check)\n:{0}%".format((1.0-clf.score(objects, classes))*100.))
#print("error percentage (overfitting check)\n:{0}%".format((1.0-clf.score(test_objects, test_classes))*100.))




