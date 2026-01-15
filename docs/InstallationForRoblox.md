# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's 2D Tensor Library" as every calculations made by our models are based on that tensor library.

| Version | Machine And Deep Learning Library (DataPredict)                                                                                                            | 2D Tensor Library (TensorL-2D)                                                                      |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Release | [DataPredict (Release Version 2.36)](https://github.com/AqwamCreates/DataPredict/blob/main/module_scripts/DataPredict%20-%20Release%20Version%202.36.rbxm) |                                                                                                     |
| Beta    | [Aqwam's Machine And Deep Learning Library](https://github.com/AqwamCreates/DataPredict/blob/main/module_scripts/AqwamMachineDeepAndReinforcementLearningLibrary.rbxm) | [Aqwam's 2D Tensor Library](https://github.com/AqwamCreates/TensorL-2D/blob/main/src/TensorL2D.lua) |

You can read the Terms And Conditions for the TensorL2D Library [here](https://github.com/AqwamCreates/TensorL-2D/blob/main/docs/TermsAndConditions.md).

To download the files from GitHub, you must click on the download button highlighted in the red box.

![Github File Download Screenshot](https://github.com/AqwamCreates/DataPredict/assets/67371914/b921d568-81b9-4f47-8a96-e0ab0316a4fe)

Then drag the files into Roblox Studio from a file explorer of your choice.

Once you put those two libraries into your game, make sure you link the DataPredict Library with the TensorL-2D Library. This can be done via setting the “AqwamTensorLibraryLinker” value (under the DataPredict) to the TensorL-2D Library.

![Screenshot 2023-12-11 011824](https://github.com/AqwamCreates/DataPredict/assets/67371914/f8dee5ef-edb0-455f-bf4a-5160ccbc35ef)

Next, we will use require() function to our machine/deep learning library:

```lua
local DataPredict = require(AqwamMachineDeepAndReinforcementLearningLibrary) 
```
