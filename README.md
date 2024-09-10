# Breast Cancer Detection using Decision Tree Classifier

This project demonstrates the use of a **Decision Tree Classifier** for predicting whether a tumor is malignant or benign using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The dataset contains features computed from a digitized image of a breast mass, which describes the characteristics of the cell nuclei present in the image.

## Dataset
The dataset is sourced from the `sklearn.datasets` package. It contains 30 features, which describe the characteristics of the tumor, such as:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- And other features related to compactness, concavity, symmetry, etc.

There are a total of 569 samples, each labeled as either:
- **0 (Malignant)**: indicating the tumor is cancerous.
- **1 (Benign)**: indicating the tumor is non-cancerous.

## Workflow

1. **Data Loading**:  
   The dataset is loaded using the `load_breast_cancer` function from `sklearn.datasets`. A pandas DataFrame is created to explore the features.

2. **Train-Test Split**:  
   The dataset is split into training and test sets using `train_test_split` from `sklearn.model_selection`. 33% of the data is reserved for testing.

3. **Model Training**:  
   A Decision Tree Classifier is trained on the training data using `DecisionTreeClassifier` from `sklearn.tree`. We set the parameter `ccp_alpha=0.01` for post-pruning to avoid overfitting.

4. **Predictions**:  
   The model predicts the outcomes on the test dataset, and probabilities are calculated using `predict_proba`.

5. **Evaluation Metrics**:  
   Various performance metrics are computed to assess the model:
   - **Accuracy Score**: Overall accuracy of the model.
   - **Confusion Matrix**: Shows the breakdown of true positives, true negatives, false positives, and false negatives.
   - **Precision**: Measures the accuracy of positive predictions.
   - **Recall**: Measures the model's ability to detect positive samples.

6. **Feature Importance**:  
   The most important features contributing to the classification decision are identified and plotted using a bar chart.

7. **Decision Tree Visualization**:  
   The decision tree is visualized using `matplotlib` and `tree.plot_tree`.

## Key Metrics

- **Accuracy**: The model achieves an accuracy score of approximately `0.93` on the test dataset.
- **Precision**: Precision score is `0.96`, indicating a high accuracy of positive predictions.
- **Recall**: Recall score is `0.97`, indicating the model's strong ability to detect positive cases.

## Visualizations

- **Feature Importance Plot**: Displays the top contributing features in predicting the tumor type.
- **Decision Tree Plot**: A graphical representation of the decision tree used by the classifier.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
