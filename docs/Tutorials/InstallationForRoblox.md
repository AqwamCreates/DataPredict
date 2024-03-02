# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Matrix Library" as every calculations made by our models are based on that matrix library.

| Version                     | Machine And Deep Learning Library (DataPredict)                                                                                                                         | Matrix Library (MatrixL)                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Release (ModuleScript)      | [DataPredict (Release Version 1.13)](https://github.com/AqwamCreates/DataPredict/blob/main/module_scripts/DataPredict%20%20-%20Release%20Version%201.13.rbxm)           |                                                                                                                    |
| Unstable (ModuleScript)     | [Aqwam's Machine And Deep Learning Library](https://github.com/AqwamCreates/DataPredict/blob/main/module_scripts/AqwamMachineAndDeepLearningLibrary.rbxm)               | [Aqwam's Matrix Library](https://github.com/AqwamCreates/MatrixL/blob/main/module_scripts/AqwamMatrixLibrary.rbxm) |

To download the files from GitHub, you must click on the download button highlighted in the red box.

![Github File Download Screenshot](https://github.com/AqwamCreates/DataPredict/assets/67371914/cdd701ba-daed-4ef4-9485-5c690077adf6)

Then drag the files into Roblox Studio from a file explorer of your choice.

Once you put those two libraries into your game, make sure you link the Machine Learning Library with the Matrix Library. This can be done via setting the “AqwamMatrixLibraryLinker” value (under the Machine Learning library) to the Matrix Library.

![Screenshot 2023-12-11 011824](https://github.com/AqwamCreates/DataPredict/assets/67371914/f8dee5ef-edb0-455f-bf4a-5160ccbc35ef)

Next, we will use require() function to our machine/deep learning library

```lua
local DataPredict = require(AqwamMachineAndDeepLearningLibrary) 
```
