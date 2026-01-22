# List Of Roblox Optimizations

## Mathematical Optimizations

* Used sufficient statistics for statistical models to avoid computationally expensive batch training.

## Roblox-Specific Optimizations

* Almost all model parameters are stored as tables of tables. This reduces the number of bytes needed to store inside Roblox's DataStores.

* All models have task.wait() per iteration to avoid eating up all computational resources.
