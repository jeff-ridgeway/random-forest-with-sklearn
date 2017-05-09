#!/usr/bin/python
'''
    Create a Random Forest model using the iris training data set
    Libraries needed include: sklearn, pandas, numpy, & pickle
    Date: 01/10/17 - 05/09/17
    Author(s): Jefferson Ridgeway, ECSU Computer Science Student
    Email: jdridgeway4@gmail.com
'''
import os;
import sys;
import math;
import numpy as np;
import pickle;
import pandas as pd;
#from sklearn import cross_validation;
from sklearn.model_selection import train_test_split;
from sklearn.tree import export_graphviz;
from sklearn.metrics import confusion_matrix, f1_score;
from sklearn.ensemble import RandomForestClassifier;

def data_split(iFile):
    '''
        Splits iris data set into test.csv and train.csv and replaces classifications with numbers
        Args:
            iFile (file_input), inputted file from the user to be split into separate .csv files
        Returns:
            None
        Raises:
            IOError
    '''
    dataset = pd.read_csv(iFile, header=None);
    #change string classifiers to numbers for iris dataset only
    dataset.iloc[:,-1] = dataset.iloc[:,-1].replace({'Iris-setosa':1});
    dataset.iloc[:,-1] = dataset.iloc[:,-1].replace({'Iris-versicolor':2});
    dataset.iloc[:,-1] = dataset.iloc[:,-1].replace({'Iris-virginica':3});
    print(list(dataset));

    num_test_size = float(input("Please input in decimal format your test size of your file: "));
    train,test = train_test_split(dataset, test_size=num_test_size);
    f1 = open('train.csv', 'w+');
    train.to_csv('train.csv')
    f1.close();
    print("Your train.csv has been created from your original file");
    f2 = open('test.csv', 'w+');
    test.to_csv('test.csv');
    f2.close();
    print("Your test.csv has been created from your original file");

def training_model():
    '''
        Uses train.csv file split from data_split() and trains the data set using
        Random Forest classifier from the sklearn library.
        Also a .p (pickle) model is created from the trained model
        Args:
            None
        Returns:
            None
        Raises:
            IOError
    '''
    if os.path.isfile('train.csv'):
        train = pd.read_csv('train.csv');
        print('The program has found your train.csv file, lets keep going!\n');
    else:
        print('The program was not able to find your file.\nPlease make sure your file is in your current directory where this program is located.');
        sys.exit();

    print(train.head());
    response = input("Is your output/Y variable at the beginning or end of the train csv file?\nType 1 for beginning, 2 for end: ");
    if response == '1':
        y_col = train.iloc[:,1];
        x_col = train.iloc[:,2:len(train.columns)-1];
    elif response == '2':
        y_col = train.iloc[:,-1];
        #print(y_col.shape);
        x_col = train.iloc[:,1:-1];
        #print(x_col.shape);

    ## training with RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100);
    rf.fit(x_col, y_col);

    ## plot randomForest
    #export_graphviz(rf);

    ## create pickle of model
    print("Creating pickle of model...");
    filename = 'randomForestmodel.p'
    pickle.dump(rf, open(filename, 'wb'));
    print("Pickle created!");

def testing_model():
    '''
        Uses test.csv file split from data_split() and tests the data set using
        the pickle model generated from training_model()
        Args:
            None
        Returns:
            y_col (single column vector), the actual y column values
            y_predict (sinlge column vector), predicted y column values using random forest model from pickle
        Raises:
            IOError
    '''
    if os.path.isfile('test.csv'):
        test = pd.read_csv('test.csv');
        print('The program has found your test.csv file, lets keep going!\n');
    else:
        print('The program was not able to find your file.\nPlease make sure your file is in your current directory where this program is located.');
        sys.exit();

    response = input("Is your output/Y variable at the beginning or end of the test csv file?\nType 1 for beginning, 2 for end: ");
    if response == '1':
        y_col = test.iloc[:,1];
        x_col = test.iloc[:,2:len(train.columns)-1];
    elif response == '2':
        y_col = test.iloc[:,-1];
        x_col = test.iloc[:,1:-1]

        ## practice for saving the x_col
        x_col.to_csv('x_col_file.csv');

    print("Loading pickle...");
    loaded_model = pickle.load(open('randomForestmodel.p', 'rb'));
    y_predict = loaded_model.predict(x_col);
    print("Successfully predicted using pickle!")
    #print(y_predict.shape);
    #print(y_col.shape);

    return(y_col, y_predict);

def accuracy_confmatrix(a1):
    '''
        Calculates the accuracy of the confusion matrix & prints out value:
        More information here: https://en.wikipedia.org/wiki/Confusion_matrix
        &
        http://www2.cs.uregina.ca/~dbd/cs831/notes/confusion_matrix/confusion_matrix.html
        Args:
            a1 (numpy multidimensional array), confusion matrix
        Returns:
            None
        Raises:
            IOError
    '''
    lengthOfArray = a1.size;
    lengthOfRow = int(math.sqrt(lengthOfArray));
    #a1 is confusion matrix
    arrSum = np.trace(a1);
    n_arrSum = 0;
    for i in range(lengthOfRow):
        for j in range(lengthOfRow):
            n_arrSum = n_arrSum + a1[i,j];

    accuracy = float(arrSum/n_arrSum);
    print("The accuracy of the confusion matrix is: %.3f\n" % accuracy);

def precision_confmatrix(a1):
    '''
        Calculates the precision of the confusion matrix and prints out value:
        More information here: https://en.wikipedia.org/wiki/Confusion_matrix
        &
        Check calculations with: http://www.marcovanetti.com/pages/cfmatrix/
        Args:
            a1 (numpy multidimensional array), confusion matrix
        Returns:
            None
        Raises:
            IOError
    '''
    lengthOfArray = a1.size;
    lengthOfRow = int(math.sqrt(lengthOfArray));
    #row in confusion matrix
    for i in range(lengthOfRow):
        #reset the row sum and max number to 0
        p_arrSum = 0;
        maxNum = 0;
        #item in column
        for j in range(lengthOfRow):
            p_arrSum = p_arrSum + a1[i,j];
            if (j != lengthOfRow - 1):
                if (a1[i,j] > a1[i,j+1]):
                    maxNum = a1[i,j];
                    #will the biggest number on the last row always be the last number in the confusion matrix?
                    #this works if the above statement is always true
            if (i == lengthOfRow -1 and j == lengthOfRow -1):
                maxNum = a1[i,j];
        precision = float(maxNum/p_arrSum);
        # print(p_arrSum);
        # print(maxNum);
        print("The precision for row %d is: %.3f\n" % (i, precision));

def f1_confmatrix(y_c, y_p):
    '''
        Calculates the f1_score of the confusion matrix & prints out value:
        More information here: https://en.wikipedia.org/wiki/Confusion_matrix
        Args:
            y_c (single column vector), the actual y column values
            y_p (sinlge column vector), predicted y column values using random forest model from pickle
            ***vectors come from testing_model()***
        Returns:
            None
        Raises:
            IOError
    '''
    f1_n1 = f1_score(y_c,y_p,average='macro');
    f1_n2 = f1_score(y_c, y_p, average='micro');

    print("The macro f1 score for the confusion matrix is %.3f" % (f1_n1));
    print("The micro f1 score for the confusion matrix is %.3f" % (f1_n2));
       
def data_evaluation(*y):
    '''
        Creates/prints confusion matrix and send confusion matrix array to the functions:
        accuracy_confmatrix(), precision_confmatrix(), & f1_confmatrix()
        More information here: https://en.wikipedia.org/wiki/Confusion_matrix
        Args:
            *y, unpacks sequence/collection of the following variables:
                y_c (single column vector), the actual y column values
                y_p (sinlge column vector), predicted y column values using random forest model from pickle
        Returns:
            None
        Raises:
            IOError
    '''
    print('The confusion matrix is as follows:\n');
    count = 0;
    for x in y:
        if count == 0:
            y_col = x;
        elif count == 1:
            y_predict = x;
        count += 1;
    print(confusion_matrix(*y));
    arr1 = confusion_matrix(*y);
    #new_arr1 = arr1.flatten();

    ## calculate the accuracy of confusion_matrix
    print('Lets get started with calculating the accuracy of the confusion matrix!');
    accuracy_confmatrix(arr1);

    ## calculate the precision of confusion_matrix
    print('Lets get started with calculating the precision of each row in the confusion matrix!');
    precision_confmatrix(arr1);

    ## calculate the f1 of confusion_matrix
    print('Lets get started with calculating the f1 value in the confusion matrix!');
    f1_confmatrix(y_col, y_predict);
def predict_model(pFile, x_col_file):
    '''
        Predict model using pickle (not functional)
        Args:
            pFile (pickle), pickle model created from training_model()
            x_col_file (file_input), x column from test.csv
        Returns:
            None
        Raises:
            IOError
    '''
    # checks first to see if pickle model is in directory same as code
    if os.path.isfile(pFile):
        print('The program has found your pickle file, lets keep going!\n');
    else:
        print('The program was not able to find your file.\nPlease make sure your file is in your current directory where this program is located.');
        sys.exit();

    if os.path.isfile(x_col_file):
        print('The program has found your X column file, lets keep going!\n');
    else:
        print('The program was not able to find your file.\nPlease make sure your file is in your current directory where this program is located.');
        sys.exit();

    x_col = pd.read_csv(x_col_file);

    print("Loading pickle...");
    loaded_model = pickle.load(open(pFile, 'rb'));
    y_predict = loaded_model.predict(x_col);
    print("Successfully predicted using pickle!")
    #print(y_predict.shape);
    #print(y_col.shape);


def main():
    userInput = input("\nLet's get started with the machine learning model!\nPlease Input your data file: ");
    data_split(userInput);
    choice = input("\nWould you like to start training?\nInput 0 for yes & 1 for no: ");
    if choice == '0':
        #direct = input("Please input your current directory: ");
        training_model();
        print("Your model is done training! Let's do some testing...\n");
        y = testing_model();
        print("\nYour model is done testing! Let's evaluate the model...\n");
        data_evaluation(*y);
    elif choice == '1':
        print('The program will now end.\n');
        sys.exit();
    else:
        print("You've inputted a wrong number and/or symbol. Let's try again!\n");
        main();

    print("Thanks for using this program! Come again!");

if __name__=='__main__':
    main();
