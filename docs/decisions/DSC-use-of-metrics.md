# DSC-0002: Mean Absolute Error
# Date: 2-11-2024
# Decision: Implementation of Mean Absolute Error metric in metric.py
# Status: Accepted
# Motivation: Providing a metric for evaluating average prediction errors
# Reason: This metric is frequently used for its simplicity and interpretability
# Limitations: Less sensitive to outliers than for example squared error metrics
# Alternatives: Root mean Squared Error metric

# DSC-0003: Root Mean Squared Error
# Date: 2-11-2024
# Decision: Implementation of Root Mean Squared Error metric in metric.py
# Status: Accepted
# Motivation: Model error units can differ from the data
# Reason: The Root Mean Squared error expresses an error in the same units as the data
# Limitations: Sensitive to outliers and higher error values
# Alternatives: Mean Squared Error or Mean Absolute Error

# DSC-0004: R-squared
# Date: 3-11-2024
# Decision: Implementation of R-squared metric in metric.py
# Status: Accepted
# Motivation: Evaluating a model's performance with statistical measures of fit
# Reason: R-squared provides the proportion of variance in the target variable
# Limitations: Limited interpretability in non-linear models
# Alternatives: Adjusted R-squared or other fit-based metrics

# DSC-0007: Macro-Average Precision
# Date: 4-11-202024
# Decision: Implementation of Macro-Average Precision metric in metric.py
# Status: Accepted
# Motivation: In multi-class classification, pre-class precision can vary
# Reason: Computes an average of per-class precision
# Limitations: Assumes equal importance across classes
# Alternatives: Micro-average precision or weighted-average precision

# DSC-0008: Macro-Average Recall
# Date: 4-11-2024 
# Decision: Implementation of Macro-Average Recall metric in metric.py
# Status: Accepted
# Motivation: In multi-class classification, recall assessment is necessary
# Reason: Computes the mean recall for each class
# Limitations: May not reflect true performance in imbalanced datasets
# Alternatives: Micro-average recall or weighted-average recall

# DSC-0018: Micro-Average Precision
# Date: 10-11-2024
# Decision: Implementation of Micro-Average Precision metric in metric.py
# Status: Accepted
# Motivation: In multi-class classification, it measures precision for every class
# Reason: The metric aggregates the true positives and false positives across classes
# Limitations: Can underrepresent minority class performances
# Alternatives: Macro-average precision

# DSC-0019: Micro-Average Recall
# Date: 10-11-2024
# Decision: Implementation of Micro-Average Recall metric in metric.py
# Status: Accepted
# Motivation: Provides a metric to assess overall recall in multi-class classification
# Reason: Micro-average recall aggregates true positives and false negatives across classes
# Limitations: May not provide insights for each class individually,
# Alternatives: Macro-average recall

# DSC-0020: Mean Absolute Percentage Error
# Date: 10-11-2024
# Decision: Implementation of Mean Absolute Percentage Error metric in metric.py
# Status: Accepted
# Motivation: To measure the average relative error in model predictions, useful for comparing across data ranges
# Reason: The metric is used mostly in regression for its interpretability as a percentage of error
# Limitations: May yield high errors for small true values
# Alternatives: Mean Absolute Error