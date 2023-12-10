# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Roblox Matrix Library" as every calculations made by our models are based on that matrix library.

| Version                     | Machine And Deep Learning Library                                                                 | Matrix Library                                                |
|-----------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Release (ModuleScript)      | [DataPredict (Release Version 1.6)](https://create.roblox.com/marketplace/asset/15552669178)      |                                                               |
| Auto Update (Package)       | [DataPredict](https://www.roblox.com/library/12727977273/DataPredict-Library)                     | [MatrixL](https://www.roblox.com/library/12728472338/MatrixL) |
| Unstable (ModuleScript)    | [Aqwam's Machine And Deep Learning Library](https://create.roblox.com/marketplace/asset/12591886004/Aqwams-Roblox-Machine-And-Deep-Learning-Library) | [Aqwam's Matrix Library](https://www.roblox.com/library/12256162800/Aqwams-Matrix-Library) |

Once you put those two libraries into your game make sure you link the Machine Learning Library with the Matrix Library. This can be done via setting the “AqwamMatrixLibraryLinker” value (under the Machine Learning library) to the Matrix Library.

![Screenshot 2023-12-11 011824](https://github.com/AqwamCreates/DataPredict/assets/67371914/f8dee5ef-edb0-455f-bf4a-5160ccbc35ef)

Next, we will use require() function to our machine/deep learning library

```lua
local DataPredict = require(AqwamMachineAndDeepLearningLibrary) 
```
