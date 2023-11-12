# Getting Started

In this library, we can customize many of our models, optimizers and others to fit our needs. This was made possible thanks to the object-orientated design of our library.

To start, we must first link our machine/deep learning library with our matrix library. However, you must use "Aqwam's Roblox Matrix Library" as every calculations made by our models are based on that matrix library.

| Version                     | Machine And Deep Learning Library            | Matrix Library     |
|-----------------------------|----------------------------------------------|--------------------|
| Release (ModuleScript)      | [DataPredict (Release Version 1.5)](https://create.roblox.com/marketplace/asset/15268337462)           |                    |
| Auto Update (Package)       | [DataPredict](https://www.roblox.com/library/12727977273/DataPredict-Library)                                      | [MatrixL](https://www.roblox.com/library/12728472338/MatrixL-Aqwams-Matrix-Library) |
| Unstable (ModuleScript)    | [Aqwam's Machine And Deep Learning Library](https://create.roblox.com/marketplace/asset/12591886004/Aqwams-Roblox-Machine-And-Deep-Learning-Library) | [Aqwam's Matrix Library](https://www.roblox.com/library/12256162800/Aqwams-Matrix-Library) |

Once you put those two libraries into your game make sure you link the Machine Learning Library with the Matrix Library. This can be done via setting the “AqwamRobloxMatrixLibraryLinker” value (under the Machine Learning library) to the Matrix Library.

![Screenshot 2023-11-12 100319](https://github.com/AqwamCreates/DataPredict/assets/67371914/d51de4c0-e2b8-4c4a-a835-12876eeb269f)

Next, we will use require() function to our machine/deep learning library

```lua
local MDLL = require(AqwamMachineAndDeepLearningLibrary) 
```
