# API Design

If you wish to create your own models and optimizers rom our library, we already have set a standard for our API design.

## Models

* All our models have train() and predict() functions.

* train() function takes in featureMatrix and labelVector (optional) in order.
  
* predict() function takes in featureMatrix or featureVector.

## Optimizers

* All our optimizers have calculate() and reset() functions.

* calculate() function takes in costFunctionDerivatives (matrix) and previousCostFunctionDerivatives (matrix) in order. It returns the adjusted costFunctionDerivatives.

* reset() does not take in any parameters.

## Regularization Objects

* All our regularization objects have calculateLossFunctionDerivativeRegularizaion() and calculateLossFunctionRegularization() functions.

* Both takes in modelParameters (matrix) and numberOfData (integer) in order.

* calculateLossFunctionDerivativeRegularizaion() returns regularization values for costFunctionDerivatives (matrix).

* calculateLossFunctionDerivativeRegularizaion() returns regularization values for modelParameters (matrix).
