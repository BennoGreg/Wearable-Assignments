#%% DEPRECATION WARNING DEACTIVATION
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import coremltools

def features_per_window(df, class_str):
    x_avg = df.iloc[:,3].mean()
    y_avg = df.iloc[:,4].mean()
    z_avg = df.iloc[:,5].mean()
    #total_avg = (x_avg + y_avg + z_avg)/3



    x_std = df.iloc[:, 3].std()
    y_std = df.iloc[:, 4].std()
    z_std = df.iloc[:, 5].std()
    #total_std = x_std + y_std + z_std

    return {'class': class_str,
                         'x_avg': x_avg,'y_avg': y_avg,'z_avg': z_avg,
            'x_std': x_std, 'y_std': y_std, 'z_std': z_std}


def extract_features(df):
    new_features = []
    windowAmount = math.floor(len(df)/200)

    for i in range(0,windowAmount*200, 200):
        extracted_data = features_per_window(df[i:i+200], df.iloc[0,1])
        new_features.append(extracted_data)

    return pd.DataFrame(new_features)


def preprocessing(df):
    walking = df[df['class'].str.contains('Walking')]
    jogging = df[df['class'].str.contains('Jogging')]
    downstairs = df[df['class'].str.contains('Downstairs')]
    upstairs = df[df['class'].str.contains('Upstairs')]
    sitting = df[df['class'].str.contains('Sitting')]
    standing = df[df['class'].str.contains('Standing')]


    jogging_non_zero = jogging[jogging['timestamp'] != 0]
    downstairs_non_zero = downstairs.loc[downstairs['timestamp'] != 0]
    sitting_non_zero = sitting.loc[sitting['timestamp'] != 0]
    walking_non_zero = walking.loc[walking['timestamp'] != 0]
    upstairs_non_zero = upstairs.loc[upstairs['timestamp'] != 0]
    standing_non_zero = standing.loc[standing['timestamp'] != 0]

    jogging_features = extract_features(jogging_non_zero)
    downstair_features = extract_features(downstairs_non_zero)
    sitting_features = extract_features(sitting_non_zero)
    walking_features = extract_features(walking_non_zero)
    upstairs_features = extract_features(upstairs_non_zero)
    standing_features = extract_features(standing_non_zero)
    new_df = pd.concat([jogging_features, downstair_features, sitting_features, walking_features, upstairs_features, standing_features])
    print(new_df)
    new_df.to_csv("extracted_data.csv",index=False)

def do_ml(df):
    data_train, data_test, target_train, target_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0],
                                                                        train_size=0.75, random_state=0)
    parameters_to_tune = {'max_features': [2 ** n for n in range(0, 3)],
                          'max_depth': [n for n in range(1, 50)]}
    dt = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters_to_tune, cv=10, scoring='accuracy')
    dt.fit(data_train, target_train)

    print("Best training parameters: " + str(dt.best_params_))
    print("Best training score: " + str(dt.best_score_))

    max_depth = dt.best_params_['max_depth']
    max_features = dt.best_params_['max_features']

    decision_tree = DecisionTreeClassifier(max_features=max_features, max_depth=max_depth)
    decision_tree.fit(data_train, target_train)
    predictions = decision_tree.predict(data_test)
    print(accuracy_score(target_test, predictions))
    print(confusion_matrix(target_test, predictions))
    print(classification_report(target_test, predictions))
    print()
    coreml_model = coremltools.converters.sklearn.convert(decision_tree,
                                                          input_features=["x_avg", "y_avg", "z_avg",  "x_std", "y_std", "z_std"],
                                                          output_feature_names="class")
    coreml_model.save('DecisionTree.mlmodel')


def analyze_data(df):
    df['class'] = df['class'].factorize()[0]
    print(df)
    correlation = df.corrwith(df['class']).round(2).sort_values(ascending=False)
    print(correlation)
    correlation_matrix = df.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.savefig("./correlation_matrix.png")
    plt.show()
    sns.pairplot(df, hue='class')
    plt.savefig("./pairplot.png")
    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv("data_f.csv")

    # preprocessing(df)

    # print('The scikit-learn version is {}.'.format(__version__))
    #
    df = pd.read_csv("extracted_data.csv")
    # analyze_data(df)
    do_ml(df)


    # print(jogging.head(10))
    # print(downstairs.head(10))
    # print(sitting.head(10))
    # print(walking.head(10))




