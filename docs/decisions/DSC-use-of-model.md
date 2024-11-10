# DSC-0005: Use of KNearestNeighbors (KNN) model
# Date: 3-11-2024
# Decision: Implementation of KNN model for classification
# Status: Accepted
# Motivation: KNN can classify data into groups with complex, non-linear boundaries
# Reason: It classifies instances by identifying the most common class among nearby data points
# Limitations: Memory intensive and can be slow on large data sets
# Alternatives: Decision Trees

# DSC-0006: Use of LassoCV model
# Date: 3-11-2024
# Decision: Implementation of LassoCV for regularized linear regression
# Status: Accepted
# Motivation: LassoCV selects important features and adresses overfitting in high-dimensional data
# Reason: LassoCV applies l1 regularization which helps with selecting significant features (shrinking some other feautre coefficients to zero), and reducing complexity
# Limitations: Can be too agressive with eliminating feautures
# Alternatives: Ridge Regression

# DSC-0009: Use of DecisionTree model
# Date: 4-11-2024
# Decision: implementation of DecisionTree model for classification
# Status: Accepted
# Motivation: The DecisionTree model provides interpretaility and transparency 
# Reason: The model splits data into branches based on feature values
# Limitations: It is prone to overfitting, which can affect generalization to new data
# Alternatives: Random Forests or Logistic Regression

# DSC-0010: Use of MultipleLinearRegression model
# Date: 4-11-2024
# Decision: Implementation of MultipleLinearRegression for regression
# Status: Accepted
# Motivation: Predicts continuous target variables based on multiple features
# Reason: The model assumes a linear relationship between features and the target and thus provides direct and interpretable relationships
# Limitations: Assumes linearity and is sensitive to multicollinearity among features
# Alternatives: Ridge regression or lasso regression

# DSC-0011: Use of Ridge model
# Date: 4-11-2024
# Decision: Implementation of Ridge for regularized linear regression
# Status: Accepted
# Motivation: Ridge minimizes overfitting in data without eliminating features
# Reason: The model uses l2 regularization th shrink large coefficients, reducing model complexity without setting coefficients to zero. Which makes it very effective for multicollinear datasets
# Limitations: Less effective at feature selection when compared to Lasso
# Alternatives: Lasso regression

# DSC-0012: Use of Logistic Regression model
# Date: 6-11-2024
# Decision: Implementation of Logistic Regression model for classification
# Status: Rejected
# Motivation: Need for a probabilistic model that efficiently handles multi-class classification problems
# Reason: Logistic Regression uses a logistic function to calculate the likelihood of belonging to a class. While efficient and interpretable, it is not appropriate for non-linear classification tasks or datasets with non-linear relationships between features and targets
# Limitations: The model struggles with non-linear data
# Alternatives: Decision Trees or KNearestNeighbors

# DSC-0016: Use of Linear SVC model
# Date: 9-11-2024
# Decision: Implementation of Linear SVC model for classification
# Status: Rejected
# Motivation: The model maximises the margin between classes in a linear classification problem
# Reason: The model works when the data is linearly seperable but may not perform with complex non-linear relationships
# Limitations: Struggles with non-linear data, and may require significant tuning of hyperparameters to avoid underfitting or overfitting
# Alternatives: Random Forest model

# DSC-0017: Use of Random Forest model
# Date: 9-11-2024
# Decision: Implementation of the Random Forest model for classification
# Status: Accepted
# Motivation: The model mitigates overfitting issues
# Reason: Random Forest combines multiple Decision Trees, the predicitons of these trees are then aggregated to produce a more accurate and generalized result
# Limitations: Computationally expensive and harder to interpret
# Alternatives: Gradient Boosting or XGBoost