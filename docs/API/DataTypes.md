# [API Reference](../API.md) - DataTypes

## FeatureMatrix / Matrix

Typically contains numbers stored in a (m x n) matrix, where m is the number of data and n is the number of features.

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

## FeatureVector / Vector

Typically contains numbers stored in a (1 x n) matrix, where n is the number of features.

### Examples

```
featureVector1 = {

  {10,2, 9}

}

featureVector2 = {

  {11, 30}

}

featureVector3 = {

  {1, 31, 123, 30, 120}

}
```

## LabelVector

Typically contains numbers stored in a (m x 1) matrix, where m is the number of data.

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

## LabelMatrix

Typically contains numbers stored in a (m x o) matrix, where m is the number of data and o is the number of classes.

### Examples

```
labelVector1 = {

  {1, 0.1, 99},
  {1, 2, 3}

}

labelVector2 = {

  {-1, 12},
  {1, 12},
  {2, 1}

}

labelVector3 = {

  {1, 0, 0, 0, 0},
  {0, 0, 0, 1, 0},
  {0, 0, 1, 0, 0},
  {0, 0, 0, 0, 1},

}
```

## TokenSequenceArray / TokenInputSequenceArray / TokenOutputSequenceArray

Contains a sequence of positive integers or zeros (a.k.a tokens) in a table. It can have any length.

### Examples

```
tokenInputSequenceArray1 = {1, 0, 30, 2}

tokenInputSequenceArray2 = {3, 4}

tokenOutputSequenceArray1 = {10, 2, 0}

tokenOutputSequenceArray2 = {1, 2, 0, 1}
```


## TableOfTokenSequenceArray / TableOfTokenInputSequenceArray / TableOfTokenOutputSequenceArray

Contains TokenInputSequenceArrays / TokenOutputSequenceArrays in a table. It can have any length.

### Examples

```
tableOfTokenInputSequenceArray1 = {

  {1, 0, 30, 2}.
  {3, 4}

}

tableOfTokenInputSequenceArray2 = {

  {1, 0, 30, 20, 10, 0, 20}

}

tableOfTokenOutputSequenceArray1 = {

  {10,2, 0, 1},
  {1,2, 0, 1}

}

tableOfTokenOutputSequenceArray2 = {

  {10},
  {1, 0, 0, 1},
  {1, 60, 1}

}
```
