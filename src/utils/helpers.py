import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(path):
    """Loads data from a file and returns a Pandas dataframe.

    :param: path: str The file path of the data files.

    :returns pandas.DataFrame The data as a Pandas dataframe.
    """
    # load train data sets
    df_demo = pd.read_excel(path + 'demo.xlsx')
    df_health = pd.read_excel(path + 'health.xlsx')
    df_habits = pd.read_excel(path + 'habits.xlsx')
    df = df_demo.merge(df_health)
    df = df.merge(df_habits)

    return df


def detect_outliers(data, method='iqr'):
    """ Detects outliers of data using the IQR,  Z-score or empirical rule method.

    :param:
    data: pandas.DataFrame - The data to be analyzed.
    method: string, optional - emp, z_score, iqr

    :returns
    pandas.DataFrame
    """

    metrics = data.select_dtypes(include=np.number).columns.tolist()
    outliers_summary = {}
    outliers = {}
    if method == 'iqr':
        """Identify the  outliers by IQR rule."""

        q25 = data.quantile(.25)
        q75 = data.quantile(.75)
        iqr = (q75 - q25)
        upper_lim = q75 + 1.5 * iqr
        lower_lim = q25 - 1.5 * iqr

        for metric in metrics:
            llim = lower_lim[metric]
            ulim = upper_lim[metric]
            outlier = [x for x in data[metric] if x < llim or x > ulim]
            if len(outlier) > 0:
                outliers[metric] = outlier
                outliers_summary[metric] = len(outlier)

        return [outliers_summary, outliers]
    elif method == 'emp':
        for metric in metrics:

            # calculate summary statistics
            data_mean, data_std = np.mean(data[metric]), np.std(data[metric])

            # identify outliers
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            outlier = [x for x in data[metric] if x < lower or x > upper]
            if len(outlier) > 0:
                outliers[metric] = outlier
                outliers_summary[metric] = len(outlier)

        return [outliers_summary, outliers]


def get_hypertension_grp(blood_pressure):
    """ Define function for grouping Diastolic BP readings into grades by blood pressure var

    :param:
    blood pressure value
    :return: string
    """

    if blood_pressure < 80:
        return 'Optimal'

    elif (blood_pressure >= 80) & (blood_pressure <= 84):
        return 'Normal'

    elif (blood_pressure >= 85) & (blood_pressure <= 89):
        return 'High_Normal'

    elif (blood_pressure >= 90) & (blood_pressure <= 99):
        return 'Grade1'

    elif (blood_pressure >= 100) & (blood_pressure <= 109):
        return 'Grade2'

    elif blood_pressure >= 110:
        return 'Grade3'


def get_HDL_cholesterol_grp(cholesterol):
    """ Define function for grouping HDL cholesterol readings into grades by High Cholesterol var

    :param cholesterol:
    :return: string
    """

    if cholesterol < 150:
        return 'Normal'

    elif (cholesterol >= 150) & (cholesterol <= 199):
        return 'Borderline_High'

    elif (cholesterol >= 200) & (cholesterol <= 499):
        return 'High'

    elif cholesterol >= 500:
        return 'Very_High'


def rpe_optimum_features_num(X_train, y_train, X_val, y_val):
    nof_list = list(range(1, 8))
    high_score = 0  # Variable to store the optimum features
    nof = 0
    score_list = []
    for n in nof_list:
        lr_model = LogisticRegression()

        rfe = RFE(estimator=lr_model, n_features_to_select=n)
        X_train_rfe = rfe.fit_transform(X=X_train, y=y_train)
        X_val_rfe_pred = rfe.transform(X_val)

        lr_model.fit(X_train_rfe, y_train)
        score = lr_model.score(X_val_rfe_pred, y_val)

        score_list.append(score)
        if score > high_score:
            high_score = score
            nof = nof_list[n]
    return nof, high_score


def generate_confusion_matrix(cf_matrix, title='', figures_path=''):
    """
    This function prints and plots the confusion matrix.
    """
    classes = ['more likely to suffer from SP', 'not likely to suffer from SP']
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=plt.cm.Blues)

    tick_marks = np.arange(len(classes)) + 0.5
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=0)
    plt.title(title + " CF_Matrix")
    plt.savefig(os.path.join(figures_path, title.replace(" ", "_") + 'cf_heatmap.png'), dpi=200)
    plt.show()


def generate_classification_report(y_val, val_pred):
    """
    This function prints classification metrics report.
    """
    print(
        '_______________________Begin Classification Report______________________')
    # print("TRAIN - Confusion Matrix: ")
    # print(confusion_matrix(y_train, train_pred))
    # print("TRAIN - Metrics Report: \n" + classification_report(y_train, train_pred))

    print("Confusion Matrix: ")
    print(confusion_matrix(y_val, val_pred))
    print("Metrics Report: \n" + classification_report(y_val, val_pred))
    print(
        '_______________________________END Report___________________________')


def perform_cross_validation(model_, X_train, y_train, cv=5):
    """Perform cross validation on train data"""
    cv_score = cross_val_score(model_, X_train, y_train, cv=cv)
    cv_score = np.sqrt(np.abs(cv_score))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" %
          (cv_score.mean(), cv_score.std(), cv_score.min(), cv_score.max()))


def model_hyperparameter_tuning(model_, grid_params, X_train, y_train, X_val, X_test, cv=5):
    grid = GridSearchCV(estimator=model_, param_grid=grid_params, scoring='f1', cv=cv, n_jobs=-1, verbose=20)

    grid.fit(X_train, y_train)
    model_grid_best = grid.best_estimator_
    val_grid_pred = model_grid_best.predict(X_val)
    test_grid_pred = model_grid_best.predict(X_test)

    return grid, val_grid_pred, test_grid_pred


def define_baseline_mode_accuracy(X_train, y_train, X_val, y_val):
    import dabl

    print('Baseline Models Accuracy: \n')
    baseline_predictions = dabl.SimpleClassifier().fit(X_train, y_train)
    baseline_accuracy = baseline_predictions.score(X_val, y_val)
    print("\nAccuracy score", baseline_accuracy)


# kaggle submission:
def export_submission(model_name, X_test, labels_test, path, columns=['PatientID', 'Disease']):
    labels = pd.DataFrame(labels_test)
    kaggle_sub = pd.concat([pd.DataFrame(X_test.index.values), labels[0]], axis=1)
    kaggle_sub.columns = columns
    kaggle_sub.to_csv(path + model_name.replace(" ", "_") + '.csv', index=False)


class Helper:
    def __init__(self):
        pass
