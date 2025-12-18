# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary Logistic Regression classifier built using scikit-learn. It predicts whether an individual’s income is greater than or less than $50K per year. Categorical features are one-hot encoded prior to training.

## Intended Use
This model is intended for educational and demonstration purposes only, specifically as an example of deploying a machine learning pipeline. It is not intended for real-world decision-making.

## Training Data
This model was trained on the U.S. Census Income dataset, which includes demographic and employment-related attributes describing individuals and their income levels. Approximately 80% of the dataset was used for training following a stratified train-test split.

## Evaluation Data
The model was evaluated on the remaining 20% of the dataset. The evaluation data was processed using the same fitted encoder and label binarizer as the training data to ensure consistency and prevent data leakage.

## Metrics
Model evaluation produced the following results on the test dataset: Precision: 0.7285, Recall: 0.6110, and F1: 0.6646. Performance was also evaluated across categorical data slices (e.g., sex, race, education) to assess how metrics vary across subgroups.

## Ethical Considerations
The dataset contains sensitive demographic attributes, and the model’s predictions may reflect existing social and economic biases present in the data. Differences in performance across categorical slices suggest the model does not perform uniformly for all groups. As such, the model should not be used in contexts where fairness, equity, or individual outcomes are critical. Any real-world application would require additional testing, bias audits, and stakeholder oversight.

## Caveats and Recommendations
This model prioritizes simplicity and reproducibility over optimization. It does not include feature scaling, advanced regularization, or bias mitigation techniques. Future improvements could include fairness-aware evaluation, alternative modeling approaches, and more rigorous validation.