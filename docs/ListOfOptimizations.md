# List Of Optimizations

## Implementation Optimizations

* Used tables of tables and external functions for efficient tensor calculations instead of space-intensive object-oriented formats.

## Mathematical Optimizations

* Used sufficient statistics for statistical models to avoid computationally expensive batch training.

* If available, the more computationally efficient formula is used instead of the original formula.

## Roblox-Specific Optimizations

* Almost all model parameters are stored as tables of tables. This reduces the number of data needed to store inside Roblox's DataStores compared to dictionaries.

* All models have task.wait() per iteration to avoid eating up all computational resources.
