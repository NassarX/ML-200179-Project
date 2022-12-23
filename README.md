# The Smith Parasite - Predictive Model
Course project of **`Machine Learning`**  course - [MDSAA-DS](www.novaims.unl.pt/MDSAA-DS) - Fall 2022

## Details of the Project

### Introduction
>A new disease has recently been discovered by Dr. Smith, in England. 
The disease has already affected more than 5000 people, with no apparent connection between them.
> The most common symptoms include fever and tiredness, but some infected people are asymptomatic. 
> Regardless, this virus is being associated with post-disease conditions such as loss of speech, confusion, chest pain and shortness of breath.
> The conditions of the transmission of the disease are still unknown and there are no certainties of what leads a patient to suffer or not from it.
> Nonetheless, some groups of people seem more prone to be infected by the parasite than others.

### Objective
This project aims to develop a predictive model that can predict if a patient will suffer, or not, from the Smith Disease. 
The model will be trained on a `dataset containing a small quantity of sociodemographic, health, and behavioral information obtained from the patients`. 
The model will be evaluated using `f1 score` of instances correctly predicted.

### Methodology
> @TODO

### Datasets
> The training set used to build the model. 
> In this set, you have the ground truth associated to each patient, 
> **if the patient has the disease (Disease = 1) or not (Disease = 0)**
> - train demo.csv - the training set for demographic data and the target train 
> - health.csv - the training set for health related data train 
> - habits.csv - the training set for habits related data

### Data Preprocessing

Before building the model, the dataset was pre-processed in the following steps:

- Data Cleansing
  - @TODO
- Feature Engineering
  - @TODO
- Feature Scaling
  - @TODO
- Feature Encoding
  - @TODO 

### Feature Selection
- Feature Selection for Numeric Values
  - @TODO
- Feature Selection for Categorical Values
  - @TODO

### Modeling

The predictive model was built using the 6 different algorithms. The following steps were taken:

[Insert any steps taken to select the model type, such as comparison to baseline models, hyperparameter tuning, etc.]
[Insert any steps taken to train the model, such as splitting the data into training and validation sets, defining the loss function, etc.]

### Evaluation

The model was evaluated using [insert evaluation metrics] on the [insert data split used for evaluation, such as test set or holdout set]. The following results were obtained:

[Insert evaluation metric 1]: [Insert result]
[Insert evaluation metric 2]: [Insert result]
[Insert evaluation metric 3]: [Insert result]

### Conclusion

The developed model achieved F1-Score 1.0 . Further improvements can be made by improving feature engineering, feature selection, using different model architectures, etc.

The predictive model developed in this project is a useful tool for predicting if patient has the disease or not based on his demographic, health and habits data. The model's accuracy demonstrates that it is able to make reliable predictions, which can be useful for a variety of applications.
### Future Work

There are several directions in which this project could be extended in the future. Some possibilities include:

- Incorporating additional data sources to improve the model's performance.
- Exploring different machine learning models to see if they perform better on the data.
- Incorporating additional features, such as more information about patient' medical history or his diet, to see if they have an impact on the model's predictions.

### Requirements

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org)
- [dabl](https://dabl.github.io/dev/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Code

Template code is provided in the `smith_parasite.ipynb` notebook file. You will also be required to use the included dataset files `datasets` folder to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project.

### Run

1. Clone the repository to your local machine.
2. In a terminal or command window, navigate to the top-level project directory `ML-200179-Project/` 
3. Run one of the following commands to execute the code:

```bash
ipython notebook smith_parasite.ipynb
```  
or
```bash
jupyter notebook smith_parasite.ipynb
```
or open with Juoyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.
