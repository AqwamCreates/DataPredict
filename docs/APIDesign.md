# API Design

If you wish to create your own models and optimizers rom our library, we already have set a standard for our API design.

## Models

* All our models have train() and predict() functions. They will be called when using some parts in "Others" section of library.

* train() function takes in featureMatrix / tableOfTokenSequenceArray  and labelVector / tableOfTokenSequenceArray  (optional for some models) in order.
  
* predict() function takes in featureMatrix or featureVector.

* The code for the models are object-oriented.

## Optimizers

* All our optimizers have calculate() and reset() functions. They will be called inside called inside our models.

* calculate() function takes in costFunctionDerivatives (matrix) and previousCostFunctionDerivatives (matrix) in order. It returns the adjusted costFunctionDerivatives.

* reset() does not take in any parameters.

* The code for the optimizers are object-oriented.

## Regularization

* All our regularization objects have calculateCostFunctionDerivativeRegularizaion() and calculateCostFunctionRegularization() functions. They will be called inside called inside our models.

* Both takes in modelParameters (matrix) and numberOfData (integer) in order. 

* calculateCostFunctionDerivativeRegularization() returns regularization values for costFunctionDerivatives (matrix).

* calculateCostFunctionRegularization() returns regularization values for modelParameters (matrix).

* The code for the regularization objects are object-oriented.

