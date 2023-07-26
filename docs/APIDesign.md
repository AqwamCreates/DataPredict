# API Design

If you wish to create your own models and optimizers from our library, we already have set a standard for our API design.

## Models

* All our models have train() and predict() functions. They will be called when using some parts in "Others" section of library.

* train() function takes in featureMatrix / tableOfTokenSequenceArray (mandatory for all models) and labelVector / tableOfTokenSequenceArray  (optional for some models) in order.
  
* predict() function takes in featureMatrix or featureVector or tableOfTokenSequenceArray.

* The code for the models are object-oriented.

## Optimizers

* All our optimizers have calculate() and reset() functions. They will be called inside called inside our models.

* calculate() function takes in learningRate (number) and costFunctionDerivatives (matrix) in order. It returns the adjusted costFunctionDerivatives.

* reset() does not take in any parameters.

* The code for the optimizers are object-oriented.

* You can get more optimizer formulas [here](https://paperswithcode.com/methods/category/stochastic-optimization).

## Regularization

* All our regularization objects have calculateRegularization() and calculateRegularizationDerivatives() functions. They will be called inside called inside our models.

* Both takes in modelParameters (matrix) and numberOfData (integer) in order.

* calculateRegularization() returns regularization values for modelParameters (matrix).

* calculateRegularizationDerivatives() returns regularization values for costFunctionDerivatives (matrix).

* The code for the regularization objects are object-oriented.

