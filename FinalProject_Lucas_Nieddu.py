#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:24:48 2020
author: Lucas Nieddu 
class: Machine Learning - Final Project
Description: ml_titanic via kaggle dataset. 
"""
# imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

#sklearn imports
#pre - processing imports 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# Model imports 
import sklearn.svm as svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Analysis imports 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier


#TensorFlow imports
from keras import Sequential
from keras.layers import Dense
import warnings


class Titanic():
    
    #Global Variables 
    global TEST_CSV, TRAIN_CSV
    TEST_CSV = 'test.csv'
    TRAIN_CSV = 'train.csv'
    
    
    #Helpfull methods 
    def load_data():
        
    #load training data
        def loadTrainData():
            try:
                train_data = pd.read_csv(TRAIN_CSV)
                print('training data Load successfull\n')
            except Exception as e:
                print('Error{}' . format(e))
            return train_data
        
        print("Loading train data...")
        train_data = loadTrainData()
        
        #Load testing data
        def loadTestData():
            try:
                test_data = pd.read_csv(TEST_CSV)
                passId = pd.DataFrame(test_data["PassengerId"])
            except Exception as e:
                print('Error{}' . format(e))
            return test_data, passId
        
        print("Loading test data...")
        test_data, passId = loadTestData()
        
        return train_data, test_data, passId
    
    #Used to create a file to hold data/plots
    global createFolder  
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)
    
    # pre- process data
    def process_data(train_data, test_data):
    
        def get_target_var():
            y_train = pd.DataFrame(train_data["Survived"])
            x_train = train_data.drop(columns=['Survived'])
            return x_train, y_train
        
        x_train, y_train = get_target_var()

        def glance_at_data():
            print('Shapes')
            print('feature matrix -->',x_train.shape)
            print('target matrix -->',y_train.shape)
        
            print('Feature info:')
            print('\nInfo of X data\n')
            print(x_train.info())
            print('\nInfo of Y data\n')
            print(y_train.info())
            print("\nNull Value counts of feature data and Target Variable....")
            print(x_train.isnull().sum())
            print(y_train.isnull().sum())
            
            
        print("Quick glance of the data...")    
        glance_at_data()
        
        #Used to view feature plots 
        def view_var(feature):
            print("Creating folder for plots...")
            createFolder('./data/')
            
            def get_plots():
                sbn.set(style = "white")
                sbn.set(style = "whitegrid", color_codes = True)
                sbn.countplot(x = feature, data = train_data, 
                              palette = "deep")
                plt.title(str(feature) + " Count")

                plt.savefig('data/' + str(feature) + '_p.png')
                plt.show()
                
                if feature != 'Survived':
                    sbn.countplot(x = feature, hue = "Survived", 
                                  data = train_data)
                    plt.title(str(feature) + " Survival Count")
                    plt.savefig('data/' + str(feature) + '_cp.png')
                    plt.show()
            get_plots()
        
        #view the features plots 
        view_var('Sex')
        view_var('Embarked')
        view_var('Survived')
        view_var('Pclass')
        view_var('SibSp')
        view_var('Parch')
        #view_var('Age')
        #view_var('Cabin')
             
        def check_missing_data():
            print('\nChecking for null values:\n')
            print('Null values for feature matrix X:')
            print(x_train.isnull().sum())
            print('\nNull values for target Y:')
            print(y_train.isnull().sum())
            
            # Age
            print("Filling in null values for feature 'Age'...")
            train_mean_age = x_train["Age"].mean()
            x_train["Age"].fillna(train_mean_age, inplace = True)
            test_mean_age = test_data["Age"].mean()
            test_data["Age"].fillna(test_mean_age, inplace = True)
            
            #Embarked
            print("\nFixing null values for 'Embarked'...\n")
            x_train["Embarked"].fillna("Q", inplace = True)
            test_data["Embarked"].fillna("Q", inplace = True)
            print(x_train.isnull().sum())
            
            #Fare
            test_mean_fare = test_data["Fare"].mean()
            test_data["Fare"].fillna(test_mean_fare, inplace = True)
        
        global feature_engineering
        def feature_engineering():
            print("Feature engineering...")
            
            
            
        print("Checking null values...")
        check_missing_data()
        
        global drop_feature
        def drop_feature(feature):
            print("Dropping feature ",feature)
            x_train.drop(columns=[str(feature)], inplace = True)
            test_data.drop(columns=[str(feature)], inplace = True)
        
        print("Dropping features names, cabin, ticket and passengerId...")
        drop_feature('Cabin')
        drop_feature('Ticket')
        drop_feature('Name')
        drop_feature('PassengerId')
        
        def encode_cat():
            labelencoder_X = LabelEncoder()    
            x_train['Embarked'] = labelencoder_X.fit_transform(x_train['Embarked'])
            x_train['Sex'] = labelencoder_X.fit_transform(x_train['Sex'])
            test_data['Embarked'] = labelencoder_X.fit_transform(test_data['Embarked'])
            test_data['Sex'] = labelencoder_X.fit_transform(test_data['Sex'])
        
        print("Encoding Categorical features...")
        encode_cat()
        print(x_train.info())
        
        #Standarize the data
        print("Standarize the data...")
        # Get column names first
        names = x_train.columns
        scaler = preprocessing.StandardScaler()
        scaled_df = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(scaled_df, columns=names)
        return x_train, y_train, test_data
    
    #Variable Selection
    def variable_selection(x_train, y_train, test_data):
        
        #Split data
        x_train,x_test,y_train,y_test=train_test_split(x_train,
                                                       y_train,test_size=0.2)
        
        #Used to find the features the best predict the target variable
        model = ExtraTreesClassifier()
        model.fit(x_train,y_train)
        print(model.feature_importances_) 
        
        #get correlations of each features in dataset
        corrmat = x_train.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(20,20))
        
        #plot heat map
        sbn.heatmap(x_train[top_corr_features].corr(),annot=True,
                    cmap="RdYlGn")
        plt.savefig('data/heatmap_1.png')
        plt.show()
        
        #plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, 
                                     index=x_train.columns)
        feat_importances.nlargest(6).plot(kind='barh')
        plt.savefig('data/feat_importance.png')
        plt.show()
        X_et = pd.DataFrame([feat_importances.nlargest(6)])
        print(X_et)
        
        #correlations of each features in dataset
        def corr_features():
            corrmat = X_et.corr()
            top_corr_features = corrmat.index
            plt.figure(figsize=(20,20))
            
            #plot heat map
            sbn.heatmap(x_train[top_corr_features].corr(),annot=True,
                    cmap="RdYlGn")
            plt.savefig('data/heatmap_2.png')
            plt.show()
        
        corr_features()
        warnings.filterwarnings("ignore")
        
        drop_feature('Embarked')
        
        return x_train, y_train, x_test, y_test, test_data
    
    # Train multiple models and perform  model analysis 
    def train_models(x_train, y_train, x_test, y_test, test_data): 
        
        
        global conf_matrix
        def conf_matrix(y_test, pred_y):
            conf_mx = confusion_matrix(y_test, pred_y)
            print("\nConfusion Matrix ...\n", conf_mx,"\n")
            print("\nHeated confusion matrix...")
            plt.matshow(conf_mx, cmap='bone')
            plt.show()

        def logistic_regression():
            print("\nLogistic Regression...")
            model = LogisticRegression(random_state=42)
            model.fit(x_train,y_train)
            #Accuracy rating
            accuracy = cross_val_score(model, x_train,
                                       y_train, cv=10, scoring = "accuracy")
            print("\nLogistic Regression Accuracy:\n", accuracy, "\n")
            pred_y = model.predict(x_test)
            report = classification_report(y_test, pred_y)
            print(report)
          
            # Confusion Matrix 
            conf_matrix(y_test, pred_y)
            return model 
        print("Logistic Regression....")
        log_reg = logistic_regression()
        
        def support_vector_machine():
            Gamma = 0.001
            C = 1
            clf = svm.SVC(kernel = 'poly' , C = C, gamma = Gamma)
            clf.fit(x_train, y_train)
            
            #Accuracy rating
            accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=10, scoring = "accuracy")
            print("\nSVM Accuracy:\n", accuracy, "\n")
            
            pred_y = clf.predict(x_test)
            report = classification_report(y_test, pred_y)
            print(report)
            
            # Confusion Matrix 
            conf_matrix(y_test, pred_y)
      
            return clf
        
        print("Supoort Vector Machine")
        svm_c = support_vector_machine()
        
        def random_forest():
            clf = RandomForestClassifier(random_state = 42)
            clf.fit(x_train, y_train)
            
            # RFC Accuracy rating 
            accuracy = cross_val_score(clf, x_train,
                                       y_train, cv=10, scoring = "accuracy")
            print("\nRandom Forest Accuracy:\n", accuracy, "\n")
            
            pred_y = clf.predict(x_test)
            report = classification_report(y_test, pred_y)
            print(report)
    
            # Confusion Matrix
            conf_matrix(y_test, pred_y)
            return clf
        
        print("Random Forest")
        rf = random_forest()
        
        def neural_network():
            print("Neural network")
            print(x_train.shape, y_train.shape)
            nn = Sequential()
            
            #First Hidden Layer
            nn.add(Dense(4, activation='relu', 
                                 kernel_initializer='random_normal', input_dim=7))
            #Second  Hidden Layer
            nn.add(Dense(4, activation='relu', 
                                 kernel_initializer='random_normal'))
            #Output Layer
            nn.add(Dense(1, activation='sigmoid', 
                                 kernel_initializer='random_normal'))
            
            #Compiling the neural network
            nn.compile(optimizer ='adam',loss='binary_crossentropy', 
                               metrics =['accuracy'])
            
            #Fitting the data to the training dataset
            nn.fit(x_train,y_train, batch_size=10, epochs=100)
            eval_model=nn.evaluate(x_train, y_train)
            print(eval_model)
            pred_y = nn.predict(x_test)
            pred_y =(pred_y > 0.5)
            conf_matrix(y_test, pred_y)
            return nn
            
        print("Neural Network...")
        nn = neural_network()
        return log_reg, svm_c, rf, nn
                        
    if __name__ == "__main__":
        
        # Load training and testing data
        print('Loading data ...')
        train_data, test_data, passId = load_data()
        #process data
        print("Processing the data....")
        x_train, y_train, test_data = process_data(train_data, 
                                                  test_data)
        
        print("Selecting Variables from pre-processed data ...")
        x_train, y_train, x_test, y_test, test_data = variable_selection(
                                                    x_train,
                                                    y_train,
                                                    test_data)
        print("Models...")
        log_reg, svm_c, rf, nn = train_models(x_train, y_train, x_test, 
                                          y_test, test_data)
        
        
        

       
        
        
        
       
        
        
        
        
        
        
        
  
        
        

        
    
       
        
   
        
        
        

        

        
        
    

        
        
        
        
        
         
    