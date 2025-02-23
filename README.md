Economic Crisis Prediction in African Countries
Project Overview
This project aims to predict the occurrence of banking crises in African countries using machine learning techniques. The dataset contains economic indicators such as GDP, inflation, and external debt, and the target variable is whether a banking crisis occurred. The goal is to predict this binary outcome using different machine learning models.

Problem Statement
The project focuses on predicting banking crises, which are crucial events that affect the stability of a country’s financial system. Accurate predictions can help policymakers and investors prepare for potential economic downturns. Using machine learning and neural network models, we can analyze various economic features to predict whether a country will experience a banking crisis.

Dataset
The dataset used for this project contains data on various economic and financial indicators for multiple African countries over several years. Features include:

Country: Name of the country.
Banking Crisis: Binary target variable indicating whether a crisis occurred (1 for a crisis, 0 for no crisis).
Economic Indicators: Features such as GDP, inflation rate, external debt, and others that influence the prediction of a banking crisis.
Models and Methods
We explored and compared multiple models, including:

Logistic Regression: A classical machine learning model used for binary classification. The logistic regression model was optimized by fine-tuning hyperparameters such as regularization strength and solver.
Neural Network Models:
Instance 1: A basic neural network with no regularization.
Instance 2: A neural network with L2 regularization and dropout to reduce overfitting.
Instance 3: A neural network using early stopping and a learning rate scheduler for optimized performance.
Instance 4: A more complex version of Instance 3 with additional layers for more feature extraction.
Findings
The models were evaluated using the following metrics:

Accuracy
F1-score
Precision
Recall
ROC-AUC
The neural network models outperformed the logistic regression model in all metrics, particularly Instance 3, which demonstrated excellent results in terms of both accuracy and F1-score.

Model	Accuracy	F1-score	Precision	Recall	ROC-AUC
Logistic Regression	97.48%	85.71%	75.00%	100.00%	99.94%
Simple Neural Network	99.37%	96.00%	92.31%	100.00%	99.94%
Neural Network Instance 1	100.00%	100.00%	100.00%	100.00%	100.00%
Neural Network Instance 2 (L2 + Dropout)	99.37%	95.65%	100.00%	91.67%	100.00%
Neural Network Instance 3 (Early Stopping + LR Scheduler)	99.37%	96.00%	92.31%	100.00%	100.00%
Which Combination Worked Better
The best-performing model combination was Neural Network Instance 1, which achieved perfect scores in all metrics. This model has a simple architecture, but its high performance suggests that the problem might be relatively easy to model with deep learning.

ML Algorithm vs Neural Network Performance
Logistic Regression: While it performed well in terms of accuracy (97.48%), it lagged behind the neural network models in terms of F1-score and recall. Logistic Regression is a relatively simple algorithm and is sensitive to feature scaling, which was addressed using StandardScaler.
Neural Networks: The neural network models significantly outperformed logistic regression, especially in terms of F1-score and precision. Models with early stopping, L2 regularization, and learning rate scheduling were particularly effective at reducing overfitting and improving generalization.
Hyperparameters for Logistic Regression
The Logistic Regression model was optimized by tuning:

C: Regularization strength, with a lower value meaning stronger regularization.
Solver: We used the liblinear solver, which works well for smaller datasets.
Folder Structure
bash
Copier
Modifier
Economic-Crisis-Prediction/
│
├── notebook.ipynb                   # Jupyter notebook containing the project code and analysis
├── saved_models/                    # Folder containing the saved models
│   ├── logistic_regression.pkl       # Optimized Logistic Regression model
│   ├── simple_nn.h5                 # Simple Neural Network model
│   ├── nn_instance1.h5              # Neural Network Instance 1
│   ├── nn_instance2.h5              # Neural Network Instance 2 (L2 + Dropout)
│   └── nn_instance3.h5              # Neural Network Instance 3 (Early Stopping + LR Scheduler)
└── README.md                        # This file
How to Run the Project
Clone the repository:

bash
Copier
Modifier
git clone https://github.com/your-username/economic-crisis-prediction.git
cd economic-crisis-prediction
Install dependencies: You can install the required dependencies by running:

bash
Copier
Modifier
pip install -r requirements.txt
Run the notebook: Open the notebook.ipynb file in Jupyter Notebook or JupyterLab and execute the cells in order to train, evaluate, and save the models.

Loading the best saved model: You can load the best model (Neural Network Instance 1) using the following code:

python
Copier
Modifier
from tensorflow.keras.models import load_model
best_model = load_model('saved_models/nn_instance1.h5')
Video Presentation
A video presentation is included, where I discuss the project, the table of results, and the differences between the machine learning algorithm and neural network performance. The camera is on, and I explain each model's results in detail.
https://www.veed.io/view/a9fc26cd-d6dd-4467-a624-c7e2fc040cec?panel=share