# Choosing The Most Effective Environment Inputs

I've commonly encounter programmers choosing inputs that led to very slow AI learning. In this tutorial, I will show you the most recommended inputs so that you can get the best out of this library.

## Relative Magnitude Over Magnitude

Relative magnitude means that the magnitude is the result from subtracting / dividing between two magnitudes. For example:

```lua

local magnitudeChange = magnitude1 - magnitude2

local magnitudeRatio = magnitude1 / magnitude2

local distance = position1 - position2

local rotationDifference = rotation1 - rotation2

local rotationValueNeededToFaceTheEnemy = math.tanh(distanceX / distanceZ) - currentRotationY

local healthChangedAmount = currentHealth - previousHealth

local healthRatio = currentHealth / maxHealth

```

Meanwhile magnitude is the value from zero. For example:

```lua

local position = primaryPart.Position

local rotation = primaryPart.Rotation

```

These two values can effect our neural networks very differently. Below, I will describe what will happens if you choose one of these values.

### Relative Magnitude

* Because the programmer handled the differences in advance, the neural network doesn't need to try to find the solution for this. 

* This will lead to more faster learning and require smaller neural network structure.

### Magnitude

* Since this is a raw value, the neural network now has to learn a solution related to output and it might not be even correct.

* This will lead to slower learning and require larger neural network structure.

## Directional Equivalents

Directional equivalents means that the values provides the same information in terms of direction. For example, let's say if we have these three values:

* distanceX

* distanceZ

* rotationY

These can be represented as:

```lua

local rotationY = math.tanh(distanceX / distanceZ)

```

Because rotationY contains those two values, distanceX and distanceZ can be removed. However, if you want the distance for rotationY, then you need to calculate the distance using both distance X and Z.

Using redundant values will cause the learning to slow down as the neural network will have to find the connection between these values that already exists.

## Normalization

Using large input values will slow down the neural network' learning speed. It is recommended to normalize your input values so that you can train your neural network faster.

You may use any kind of normalization techniques, including:

* Z-Score Normalization (Standardization)

* Minimum-Maximum Scaling

## Conclusion

So you have to keep these three in mind:

* Relative Magnitude Over Magnitude

* Directional Equivalents

* Normalization

Once you master those three, you can start seeing your AI learn much more faster.

That's all for today!
