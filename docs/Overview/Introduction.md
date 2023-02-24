# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Roblox Matrix Library" as every calculations made by our models are based on that matrix library.

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

I will give you the codes for the featureMatrix and the labelVector for you to practice.

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
