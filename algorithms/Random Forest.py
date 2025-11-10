import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Global variables
train_path = '../data/train.csv'
test_path = '../data/test.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Plot variables
gender_plot_colors = ['#3a89de', '#96208b']
class_plot_colors = ['#ff0000', '#ff8000', '#ffc400']
embarked_plot_colors = ['#2eb847', '#5dc0c7', '#006eff']
survived_plot_colors = ['#c22b2b', '#6c2bc2']


def loead_data(train_file_path, test_file_path):
    print(f"Reading training file located at {train_path}...")
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)
    return df_train, df_test


def clean_data(df):
    print("Cleaning data...")
    # Drop duplicates
    df = df.drop_duplicates()
    return df


def encode_categorical(df):
    print("Encoding categorical values...")
    # Encode

    # Sex

    # Age


    # Embarked

    return df


def exploratory_analysis(train, test):
    print("Performing exploratory analysis...")
    # Dataset analysis
    print("Dataset analysis...")
    # Shape
    print("Shape : ")
    print(f"Train Shape : {train.shape}")
    print(f"Test Shape : {test.shape}")
    # Describe
    print("Describe : ")
    print(f"Train Shape : {train.describe}")
    print(f"Test Shape : {test.describe}")

    # For all of these, plot for total, survived, and died
    print("Data analysis...")

    # Create dataframes of train with only survived and only died
    df_train_survived, df_train_died = train, train
    df_train_survived = df_train_survived[df_train_survived['Survived'] == 1]
    df_train_died = df_train_died[df_train_died['Survived'] == 0]

    # Distribution of gender
    print("Distribution of gender")
    print("Total")
    gender_counts_total = train['Sex'].value_counts()
    male_rate_total = gender_counts_total['male'] / len(train) * 100
    female_rate_total = gender_counts_total['female'] / len(train) * 100
    print(f"Male : {gender_counts_total['male']} ({male_rate_total:.1f}%)")
    print(f"Female : {gender_counts_total['female']} ({female_rate_total:.1f}%)")
    gender_rate_data_total = [male_rate_total, female_rate_total]
    gender_labels_total = ['Male', 'Female']

    print("Survived")
    gender_counts_survived = df_train_survived['Sex'].value_counts()
    male_rate_survived = gender_counts_survived['male'] / len(train) * 100
    female_rate_survived = gender_counts_survived['female'] / len(train) * 100
    print(f"Male : {gender_counts_survived['male']} ({male_rate_survived:.1f}%)")
    print(f"Female : {gender_counts_survived['female']} ({female_rate_survived:.1f}%)")
    gender_rate_data_survived = [male_rate_survived, female_rate_survived]
    gender_labels_survived = ['Male', 'Female']

    print("Died")
    gender_counts_died = df_train_died['Sex'].value_counts()
    male_rate_died = gender_counts_died['male'] / len(train) * 100
    female_rate_died = gender_counts_died['female'] / len(train) * 100
    print(f"Male : {gender_counts_died['male']} ({male_rate_died:.1f}%)")
    print(f"Female : {gender_counts_died['female']} ({female_rate_died:.1f}%)")
    gender_rate_data_died = [male_rate_died, female_rate_died]
    gender_labels_died = ['Male', 'Female']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].pie(gender_rate_data_total, labels=gender_labels_total, colors=gender_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Total")
    axes[1].pie(gender_rate_data_survived, labels=gender_labels_survived, colors=gender_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Survived")
    axes[2].pie(gender_rate_data_died, labels=gender_labels_died, colors=gender_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[2].set_title("Died")
    fig.suptitle("Gender Distribution", fontsize=16, fontweight='bold')
    plt.show()

    # Distribution of class
    print("Distribution of class")
    print("Total")
    class_counts_total = train['Pclass'].value_counts()
    first_class_rate_total = class_counts_total[1] / len(train) * 100
    second_class_rate_total = class_counts_total[2] / len(train) * 100
    third_class_rate_total = class_counts_total[3] / len(train) * 100
    print(f"First : {class_counts_total[1]} ({first_class_rate_total:.1f}%)")
    print(f"Second : {class_counts_total[2]} ({second_class_rate_total:.1f}%)")
    print(f"Third : {class_counts_total[3]} ({third_class_rate_total:.1f}%)")
    class_rate_data_total = [first_class_rate_total, second_class_rate_total, third_class_rate_total]
    class_labels_total = ['First', 'Second', 'Third']

    print("Survived")
    class_counts_survived = df_train_survived['Pclass'].value_counts()
    first_class_rate_survived = class_counts_survived[1] / len(train) * 100
    second_class_rate_survived = class_counts_survived[2] / len(train) * 100
    third_class_rate_survived = class_counts_survived[3] / len(train) * 100
    print(f"First : {class_counts_survived[1]} ({first_class_rate_survived:.1f}%)")
    print(f"Second : {class_counts_survived[2]} ({second_class_rate_survived:.1f}%)")
    print(f"Third : {class_counts_survived[3]} ({third_class_rate_survived:.1f}%)")
    class_rate_data_survived = [first_class_rate_survived, second_class_rate_survived, third_class_rate_survived]
    class_labels_survived = ['First', 'Second', 'Third']

    print("Died")
    class_counts_died = df_train_died['Pclass'].value_counts()
    first_class_rate_died = class_counts_died[1] / len(train) * 100
    second_class_rate_died = class_counts_died[2] / len(train) * 100
    third_class_rate_died = class_counts_died[3] / len(train) * 100
    print(f"First : {class_counts_died[1]} ({first_class_rate_died:.1f}%)")
    print(f"Second : {class_counts_died[2]} ({second_class_rate_died:.1f}%)")
    print(f"Third : {class_counts_died[3]} ({third_class_rate_died:.1f}%)")
    class_rate_data_died = [first_class_rate_died, second_class_rate_died, third_class_rate_died]
    class_labels_died = ['First', 'Second', 'Third']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].pie(class_rate_data_total, labels=class_labels_total, colors=class_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Total")
    axes[1].pie(class_rate_data_survived, labels=class_labels_survived, colors=class_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Survived")
    axes[2].pie(class_rate_data_died, labels=class_labels_died, colors=class_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[2].set_title("Died")
    fig.suptitle("Class Distribution", fontsize=16, fontweight='bold')
    plt.show()

    # Distribution of age
    print("Distribution of age")
    # histogram
    print("Total")

    print("Survived")

    print("Died")

    # Distribution of Embarked
    print("Distribution of embarked")
    print("Total")
    embarked_counts_total = train['Embarked'].value_counts()
    cherbourg_embarked_rate_total = embarked_counts_total['C'] / len(train) * 100
    queenstown_embarked_rate_total = embarked_counts_total['Q'] / len(train) * 100
    southampton_embarked_rate_total = embarked_counts_total['S'] / len(train) * 100
    print(f"Cherbourg : {embarked_counts_total['C']} ({cherbourg_embarked_rate_total:.1f}%)")
    print(f"Queenstown : {embarked_counts_total['Q']} ({queenstown_embarked_rate_total:.1f}%)")
    print(f"Southampton : {embarked_counts_total['S']} ({southampton_embarked_rate_total:.1f}%)")
    embarked_rate_data_total = [cherbourg_embarked_rate_total, queenstown_embarked_rate_total,
                                southampton_embarked_rate_total]
    embarked_labels_total = ['Cherbourg', 'Queenstown', 'Southampton']

    print("Survived")
    embarked_counts_survived = df_train_survived['Embarked'].value_counts()
    cherbourg_embarked_rate_survived = embarked_counts_survived['C'] / len(train) * 100
    queenstown_embarked_rate_survived = embarked_counts_survived['Q'] / len(train) * 100
    southampton_embarked_rate_survived = embarked_counts_survived['S'] / len(train) * 100
    print(f"Cherbourg : {embarked_counts_survived['C']} ({cherbourg_embarked_rate_survived:.1f}%)")
    print(f"Queenstown : {embarked_counts_survived['Q']} ({queenstown_embarked_rate_survived:.1f}%)")
    print(f"Southampton : {embarked_counts_survived['S']} ({southampton_embarked_rate_survived:.1f}%)")
    embarked_rate_data_survived = [cherbourg_embarked_rate_survived, queenstown_embarked_rate_survived,
                                   southampton_embarked_rate_survived]
    embarked_labels_survived = ['Cherbourg', 'Queenstown', 'Southampton']

    print("Died")
    embarked_counts_died = df_train_died['Embarked'].value_counts()
    cherbourg_embarked_rate_died = embarked_counts_died['C'] / len(train) * 100
    queenstown_embarked_rate_died = embarked_counts_died['Q'] / len(train) * 100
    southampton_embarked_rate_died = embarked_counts_died['S'] / len(train) * 100
    print(f"Cherbourg : {embarked_counts_died['C']} ({cherbourg_embarked_rate_died:.1f}%)")
    print(f"Queenstown : {embarked_counts_died['Q']} ({queenstown_embarked_rate_died:.1f}%)")
    print(f"Southampton : {embarked_counts_died['S']} ({southampton_embarked_rate_died:.1f}%)")
    embarked_rate_data_died = [cherbourg_embarked_rate_died, queenstown_embarked_rate_died,
                               southampton_embarked_rate_died]
    embarked_labels_died = ['Cherbourg', 'Queenstown', 'Southampton']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].pie(embarked_rate_data_total, labels=embarked_labels_total, colors=embarked_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Total")
    axes[1].pie(embarked_rate_data_survived, labels=embarked_labels_survived, colors=embarked_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Survived")
    axes[2].pie(embarked_rate_data_died, labels=embarked_labels_died, colors=embarked_plot_colors, autopct='%1.1f%%', startangle=90)
    axes[2].set_title("Died")
    fig.suptitle("Embarked Distribution", fontsize=16, fontweight='bold')
    plt.show()

    # Distribution of survival
    print("Distribution of survival")
    survived_counts = train['Survived'].value_counts()
    survived_rate = survived_counts[1] / len(train) * 100
    died_rate = survived_counts[0] / len(train) * 100
    print(f"Survived : {survived_counts[1]} ({survived_rate:.1f}%)")
    print(f"Died : {survived_counts[0]} ({died_rate:.1f}%)")
    survived_rate_data = [survived_rate, died_rate]
    survived_labels = ['Survived', 'Died']
    plt.pie(survived_rate_data, labels=survived_labels, colors=survived_plot_colors, autopct='%1.1f%%', startangle=90)
    plt.title("Global survival rate", fontsize=16, fontweight='bold')
    plt.show()


def hyperparameter_tuning(X, y):
    print("Performing hyperparameter finetuning...")
    best_params = None

    # param_grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # random_search
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # selection of best parameters
    best_params = {
        'n_estimators': best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'min_samples_split': best_params['min_samples_split'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'bootstrap': best_params['bootstrap']
    }

    return best_params


def train_model(X, y, best_params):
    print("Training model...")
    
    # predict
    model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    model.fit(X, y)
    print("Model trained.")

    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model....")
    # accuracy

    # precision

    # recall

    # f-1 score

    # ROC Curve

    # Confusion matrix


def pipeline(train_file_path, test_file_path, target_column):
    # Load and clean data
    df_train, df_test = loead_data(train_file_path, test_file_path)
    df_train = clean_data(df_train)

    # Encode categorical variables
    df_train = encode_categorical(df_train)

    # Exploratory analysis
    exploratory_analysis(df_train, df_test)

    X = df_train.drop(columns=[target_column])
    y = df_train[target_column]

    # Performe hyperparameter tuning
    best_params = hyperparameter_tuning(X, y)
    print("Best Hyperparameters :", best_params)

    # Train final model
    model = train_model(X, y, best_params)

    # Evaluate model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    evaluate_model(model, X_test, y_test)


pipeline(train_path, test_path, "Survived")
