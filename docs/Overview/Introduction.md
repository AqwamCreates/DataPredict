In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

# Getting Started

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Roblox Matrix Library" as every calculations made by our models are based on that matrix library.

Next, we will use require() function to our machine/deep learning library

```

local MDLL = require(AqwamRobloxMachineAndDeepLearningLibrary) 


```

# Creating A Machine/Deep Learning Model Object

For our first model, we will use "LogisticRegression". We will create a new "LinearRegression" model object using new(). 

```
local LogisticRegression = MDLL.Models.LogisticRegression

local LogisticRegressionModel = LogisticRegression.new()

```

Although the new() can take in a number of arguments, we will use the default values provided by the library to simplify our introduction. You can see what different models takes as their arguments in the API Reference.

# Training Our Model

```


```

