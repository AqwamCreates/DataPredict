It is important for us to check the accuracy of our model. Hence, in this tutorial we will show you on how to check your model's accuracy.

# Getting Started

First, we need to get one of our functions from the library.

```
local ModelChecking = MDLL.Others.ModelChecking
```

# Testing For Accuracy

Right now, we will test our LogisticRegression model accuracy. Since Logistic regression is a classification algorithm, we will apply testClassificationModel(). We will also provide featureMatrix and labelVector as well.

```
local accuracy = ModelChecking:testClassificationModel(MachineLearningModel, featureMatrix, labelVector)
```

The above function will generate the accuracy of the model by comparing the predicted output made by the model and the actual value.

Then use print() to see the accuracy.

```
print(accuracy)
```

That's all for today!
