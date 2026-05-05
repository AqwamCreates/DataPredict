# What Is Data Transformation

Data transformation is basically means that we add, delete, merge or modify the values inside our dataset so that our models can train and predict better.

It technically allows you to use models that are not designed for the original dataset, provided that you maintain consistency on how you handle the modified dataset.

# Redudant Models

This may be surprising to you, but most of the models that handles regression are actually redundant. Though, those models were added for users' convenience as well as "fluffing up my model count".

To show what I mean, I'll generate a dataset that best captures this.

```lua

local featureMatrix = {

  {1},
  {2},
  {3}
  {4}
  {5}
  {6}

}
