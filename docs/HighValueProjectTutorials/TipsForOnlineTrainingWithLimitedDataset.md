# Tips For Online Training With Limited Dataset

## Avoid Using Z-Score And Minimum-Maximum Normalization

Z-Score and minimum-maximum normalization have this issue where the data distribution changes when you use different datasets to train the same models.

As such, we would recommend to scale the values relative to other values when possible. Below, we will show you how it can be done.

```lua
x = (value1 - value2) / (value1 + value2)
```
