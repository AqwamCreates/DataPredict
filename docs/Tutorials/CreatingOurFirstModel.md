# Creating A Machine/Deep Learning Model Object

For our first model, we will use "LogisticRegression". We will create a new "LogisticRegression" model object using new(). 

```lua
local LogisticRegression = DataPredict.Models.LogisticRegression

local LogisticRegressionModel = LogisticRegression.new()
```

Although the new() can take in a number of arguments, we will use the default values provided by the library to simplify our introduction. You can see what different models takes as their arguments in the API Reference. You can also change them at anytime you want using setParameters() function.

# Training Our Model

To train our model, we need to supply two things: featureMatrix and labelVector. 

* For the feature matrix, the rows are the individual data and the columns are the features for that particular data. 

* For the labelVector, the rows are the number of values and the columns are the values that have certain relationship to that individual data.

I will give you the codes for the featureMatrix and the labelVector for you to practice. You can see that if the data contains 0 or greater, it will result to 1. Otherise, the value is 0.

```lua
local featureMatrix = {
	
	{1, 0,  0},
	{1, 10, 2},
	{1, -3, -2},
	{1, -12, -22},
	{1, 2,  2},
	{1, 1,  1},
	{1,-11, -12},
	{1, 3,  3},
	{1, -2, -2},

}

-- Value of 1 is added at first column for bias.

local labelVectorLogistic = {
	
	{1},
	{1},
	{0},
	{0},
	{1},
	{1},
	{0},
	{1},
	{0}
	
}
```

With our featureMatrix and labelVector in place, we will supply them to our model's train() function.

```lua
LogisticRegressionModel:train(featureMatrix, labelVectorLogistic)
```

Once you run the function, the model will generate its model parameters. However, during your training, your model might go to unusual cases and may need to adjust certain parameters for our model. We will cover this in the next section.

In addition, not all models require labelVector. This is mainly true for our clustering models such as "KMeans". So take note of that.

# Training Cases

When training the model, the cost of the training is printed out by default. Under the normal case, the cost would follow these pattern in order:

1. Steadily increasing and decreasing (optional)

2. Steadily decreasing

3. Stabilizes or very small increase and decrease

Sometimes, our models can run into trouble due to a number of reasons. It may be because of the parameters we given or it is the hardware limitations.

## Case 1 - Cost Printing Out "nan"

It means that during training, the calculations may have resulted in either arithmetic underflow or overflow. To fix this, limit the max number of iterations or set a target cost. This is a hardware limitation issue.

Another reason is that the calculated model parameters contain infinity values. This can be solved by feature scaling. This is issue is due to the some of the dataset values are too large to be calculated by the model.

This case can also happen if any of the internal model components are calculating numbers with not a number. This issue can propagate to final output without raising any suspicion. If the "nan" value shows up after the first or second iterations and you tried to solve based on the information given above, then you need to contact me regarding this issue.

## Case 2 - Cost Printing Out "inf"

It means that during training, the model is no longer "learning" but does the complete opposite. When attempting to predict using this model, it is highly likely that you will get wrong prediction. To fix this, use an optimizer or adjust the parameters. This is a parameter issue.

# Predicting Using Our Models

To predict, we will use predict() function for our model. We will then supply data to the model so that it can predict the value.

```lua
local predictedVector = LogisticRegressionModel:predict(testData)
```

I will give you a test data for you to use. The value of prediction should be 1 and 0.

```lua
local testData = {

	{1, 90, 32}
	{1, -120, -41}

}

local predictedVector = LogisticRegressionModel:predict(testData)

local value1 = predictedVector[1][1]

local value2 = predictedVector[2][1]

print(value1) -- This is 1.

print(value2) -- This is 0.
```

# And Finally...

Don't forget to leave a like in Roblox's DevForum and the Github's repository if you find this information useful! 

Since you read the whole introduction, I recommend you to have a look at these to further your knowledge.

* [Saving And Loading Model Parameters](SavingAndLoadingModelParameters.md)

* [Using Model Checker](UsingModelChecker.md)

* [Using Optimizers](UsingOptimizers.md)

* [Using Gradient Descent Modifiers](UsingGradientDescentModifiers.md)

* [Using Regularization](UsingRegularization.md)

* [Using Neural Networks Part 1](UsingNeuralNetworksPart1.md)
