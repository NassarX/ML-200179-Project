# Data-Driven Predictive Modeling of the Smith Disease.
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
To develop a predictive model for the Smith Disease, I've followed the steps to:

1. Collect and preprocess the data: Begin by collecting a dataset of sociodemographic, health, and behavioral information about patients with the Smith Disease. This may include data on factors such as age, gender, lifestyle, preexisting health conditions, and other relevant variables. Preprocess the data to ensure that it is clean and ready for modeling.
2. Explore and visualize the data: Next, explore the data to gain a better understanding of the characteristics of the patients and any potential relationships between the variables. Use visualization techniques such as histograms, scatter plots, and box plots to visualize the data and identify patterns and trends.
3. Select and prepare the features: Select the features that you believe are most relevant for predicting the likelihood of a patient suffering from the Smith Disease. Prepare the features for modeling by scaling or normalizing the data as needed.
4. Split the data into training and test sets: Divide the prepared data into a training set and a test set. The training set will be used to fit the model, while the test set will be used to evaluate the model's performance.
5. Train the model: Use the training set to train a predictive model using a suitable machine learning algorithm. There are many algorithms to choose from, including decision trees, random forests, support vector machines, and neural networks. Select the algorithm that performs best on the training set, based on metrics such as accuracy or f1 score.
6. Evaluate the model: Use the test set to evaluate the performance of the trained model. Calculate the f1 score of correctly predicted instances to determine the model's accuracy. If the model's performance is not satisfactory, consider adjusting the model or trying a different algorithm.
7. Fine-tune the model: Once you have identified a model that performs well on the test set, fine-tune the model by adjusting its hyperparameters or adding additional features as needed.
8. Validate the model: Finally, validate the model's performance on a separate dataset to ensure that it generalizes well to new data. If the model performs well on the validation set, you can consider using it to make predictions on new cases of the Smith Disease.

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

The predictive model was built using the 6 different algorithms. Almost all the algorithms follow this pattern:

- Define the classifier
- Fit the model with the train set
- Predict on the test set, using the fitted classifier
- Compute performances of the model, showing some statistics and metrics

### Evaluation

The model was evaluated using F1 Score on the test set. The following results were obtained:

- Decision Tree: 0.99
- XGBoost: 0.996

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
