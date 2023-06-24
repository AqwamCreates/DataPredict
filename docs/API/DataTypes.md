# [API Reference](../../API.md) - DataTypes

## FeatureMatrix

Typically contains numbers stored in a (m x n) matrix.

### Examples

```
featureMatrix1 = {

  {10,2, 9},
  {1,3,6}

}

featureMatrix2 = {

  {10, 2},
  {1, 3},
  {11, 30}

}

featureMatrix2 = {

  {10, 23, 30, 111, 320},
  {1, 31, 123, 30, 120},
  {11, 30, 1234, 123, 0},
  {11, 30, 123, 323, 12}

}
```

## LabelVector

Typically contains numbers stored in a (m x 1) matrix.

### Examples

```
labelVector1 = {

  {10},
  {1}

}

labelVector2 = {

  {-1},
  {1},
  {1}

}

labelVector3 = {

  {1},
  {1},
  {0},
  {0},

}
```
