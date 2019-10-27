#THE MANUAL METHOD

import numpy as np

#function to get the value of our data
def log_odds(features,coefficients,intercept):
	return np.dot(features,coefficients) + intercept

#this function returns a value form 0-1 we take this as the probablity
def sigmoid(z):
	denominator = 1 + np.exp(-z)
	return 1 / denominator

def predict_class(featues,coefficients,intercept,threshold):
  calculated_log_odds = log_odds(featues,coefficients,intercept)
  probablities = sigmoid(calculated_log_odds)
  return np.where(probablities >= threshold ,1 ,0)

#predicting some unknow
final_result  = predict_class(hours_studied,calculated_coefficients,intercept,0.5)
print(final_result)


#SHORTCUT METHOD USING INBUILT FUNCTIONS
import numpy as np
from sklearn.linear_model import LogisticRegression

#get all the data

#model is a LogisticRegression object
model = LogisticRegression()
model.fit(values,result)

# Save the model coefficients and intercept here
coefficients = model.coef_
intercept = model.intercept_

# Predict the probabilities of matching the results
passed_predictions = model.predict_proba(test_values)

# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(new_values,results_for_new_values)

# Predict whether result comes as expected
passed_predictions_2 = model_2.predict(test_new_values)