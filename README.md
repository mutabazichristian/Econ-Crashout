Here's a human-readable "README" for your summative project on **Economic Crisis Prediction in African Countries**:

---

# **Economic Crisis Prediction in African Countries**

## **Project Overview**

This project explores the prediction of banking crises in African countries using machine learning models. The aim is to understand and predict the occurrence of banking crises using various economic and social indicators. The models use different machine learning techniques, including traditional models like **Logistic Regression** and deep learning models such as **Neural Networks**.

## **Objective**

The main goal is to predict the occurrence of a banking crisis in a given country based on various input features. The project aims to:
- Predict the occurrence of a banking crisis (binary classification: crisis or no crisis).
- Evaluate multiple machine learning models to compare performance.
- Use different approaches like classical machine learning models and neural networks to understand their strengths and weaknesses.

## **Dataset**

The dataset used in this project is called `african_crises.csv`, which contains economic and financial data from multiple African countries over several years. Key features in the dataset include:
- **Country**: The country being analyzed.
- **Banking Crisis**: Whether a banking crisis occurred (target variable: `0` for no crisis, `1` for crisis).
- **Economic and Social Indicators**: A variety of features, such as GDP, inflation rates, and external debt, that can influence the occurrence of a banking crisis.

## **Models Used**

Several models were trained and evaluated:
1. **Logistic Regression**: A classical machine learning model used for binary classification.
2. **Simple Neural Network**: A basic feed-forward neural network for predicting the banking crisis.
3. **Multiple Neural Network Instances**:
   - Instance 1: A neural network with increased complexity.
   - Instance 2: A neural network with **L2 regularization** and **dropout** to prevent overfitting.
   - Instance 3: A neural network with **early stopping** and **learning rate scheduler** to optimize performance.

### **Model Performance**

Each model was evaluated on the following metrics:
- **Accuracy**
- **F1-score**
- **Precision**
- **Recall**
- **ROC-AUC**

The neural network models, particularly those with early stopping and regularization, showed significantly better performance, with Instance 3 achieving perfect results.

## **Preprocessing**

Key preprocessing steps included:
- **Creating dummy variables** for categorical features (e.g., countries).
- **Mapping the target variable** (`banking_crisis`) to binary values.
- **Scaling features** using **StandardScaler** to ensure consistent input for machine learning models.

## **Results**

The evaluation metrics show that the neural network models significantly outperform logistic regression in terms of precision and recall. The models have been saved for future use, and their results can be reviewed through a comparison chart of metrics.

## **Saved Models**

All trained models have been saved for future use:
- **Logistic Regression**: Saved as a `.pkl` file.
- **Neural Networks**: Saved as `.h5` files.

These models are stored in the `saved_models/` directory and can be loaded for further analysis or deployment.

## **Folder Structure**

```
Economic-Crisis-Prediction/
│
├── african_crises.csv         # The dataset
├── saved_models/              # Folder containing saved models
│   ├── logistic_regression.pkl # Logistic Regression model
│   ├── simple_nn.h5           # Simple Neural Network model
│   ├── nn_instance1.h5        # Neural Network Instance 1
│   ├── nn_instance2.h5        # Neural Network Instance 2
│   └── nn_instance3.h5        # Neural Network Instance 3
├── main.py                    # Python script for running the models and evaluation
└── README.md                  # This file
```

## **How to Run the Project**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/economic-crisis-prediction.git
   cd economic-crisis-prediction
   ```

2. **Install dependencies**:
   You can create a virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:
   To train and evaluate the models, simply run:
   ```bash
   python main.py
   ```

   This will train the models and display the results.

## **Future Work**

- **Further Model Tuning**: Additional optimization techniques, such as hyperparameter tuning, could be explored.
- **Data Augmentation**: More data could be added to improve the robustness of the model.
- **Deployment**: The trained models can be deployed to make real-time predictions.

## **Acknowledgements**

- **Dataset**: Data sourced from publicly available economic indicators for African countries.
- **Libraries**: This project uses several libraries, including **TensorFlow**, **scikit-learn**, and **pandas**.

---

Let me know if you need any further modifications!