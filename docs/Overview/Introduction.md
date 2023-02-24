# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Roblox Matrix Library" as every calculations made by our models are based on that matrix library.

* [Aqwam's Machine And Deep Learning Library](https://create.roblox.com/marketplace/asset/12591886004/Aqwams-Roblox-Machine-And-Deep-Learning-Library)

* [Aqwam's Roblox Matrix Library](https://www.roblox.com/library/12256162800/Aqwams-Roblox-Matrix-Library)

Once you put those two libraries into your game make sure you link the Machine Learning Library with the Matrix Library. This can be done via setting the “AqwamRobloxMatrixLibraryLinker” value (under the Machine Learning library) to the Matrix Library.

![c](https://user-images.githubusercontent.com/67371914/221095215-d5df15ad-5b2c-4bb5-8a78-40911edd482a.PNG)


Next, we will use require() function to our machine/deep learning library

```
local MDLL = require(AqwamRobloxMachineAndDeepLearningLibrary) 
```

# Creating A Machine/Deep Learning Model Object

For our first model, we will use "LogisticRegression". We will create a new "LogisticRegression" model object using new(). 

```
local LogisticRegression = MDLL.Models.LogisticRegression

local LogisticRegressionModel = LogisticRegression.new()
```

Although the new() can take in a number of arguments, we will use the default values provided by the library to simplify our introduction. You can see what different models takes as their arguments in the API Reference. You can also change them at anytime you want using setParameters() function.

# Training Our Model

To train our model, we need to supply two things: featureMatrix and labelVector. For the feature matrix, the rows are the individual data and the columns are the features for that particular data. For the labelVector, the rows are the values that have certain relationship to that individual data.

I will give you the codes for the featureMatrix and the labelVector for you to practice. You can see that if the data contains 0 or greater, it will result to 1. Otherise, the value is 0.

```
local featureMatrix = {
	
	{ 0,  0},
	{10, 2},
	{-3, -2},
	{-442, -22},
	{ 2,  2},
	{ 1,  1},
	{-11, -22},
	{ 3,  3},
	{-2, -2},

}

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

```
LogisticRegressionModel:train(featureMatrix, labelVectorLogistic)
```

Once you run the function, the model will generate its model parameters. However, during your training, your model might go to unusual cases and may need to adjust certain parameters for our model. We will cover this in the next section.

In addition, not all models require labelVector. This is mainly true for our clustering machine/deep learning models such as "KMeans". So take note of that.

# Training Cases

When training the data, the cost of the training is printed out by default. Under the normal case, the cost would follow these pattern in order:

1. Steadily increasing and decreasing (optional)

2. Steadily decreasing

3. Stabilizes Or very small increase and decrease

Sometimes, our models can run into trouble due to a number of reasons. It may be because of the parameters we given or it is the hardware limitations.

## Case 1 - Cost Printing Out "nan"

It means that during training, the calculations may have resulted in either arithmetic underflow or overflow. To fix this, limit the max number of iterations or set a target cost. This is a hardware limitation issue.

## Case 2 - Cost Printing Out "inf"

It means that during training, the model is no longer "learning" but does the complete opposite. When attempting to predict using this model, it is highly likely that you will get wrong prediction. To fix this, use an optimizer or adjust the parameters. This is a parameter issue.

# Predicting Using Our Models

To predict, we will use predict() function for our model. We will then supply data to the model so that it can predict the value.

```
local predictedValue = LogisticRegressionModel:predict(testData)
```

I will give you a test data for you to use. The value of prediction should be 1.

```
local testData = {

	{90, 90}
	
}
```

# And Finally...

Since you read the whole introduction, I recommend you to have a look at these to further your knowledge.

* [Using Model Checking](UsingModelChecking.md)

* [Using Optimizers](UsingOptimizers.md)

* [Using Gradient Descent Modes](UsingGradientDescentModes.md)
