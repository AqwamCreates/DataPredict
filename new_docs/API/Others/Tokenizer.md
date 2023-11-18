# [API Reference](../../API.md) - [Others](../Others.md) - Tokenizer

Tokenizer is used to tokenize items such as text, number and so on. It also includes a number of useful functions for converting between tokens and items.

## Constructors

### new()

Creates a new tokenizer object.

```
Tokenizer.new(tokenizedItemArray: []): TokenizerObject
```

#### Parameters:

* tokenizedItemArray: An array containing all the items. The position of the items in the array indicates the token number. The input for this argument is optional.

#### Returns:

* TokenizerObject: An object that allows the conversion of tokens and items.

## Functions

### addItem()

Adds an item to be tokenized. If the item already exists, then it will ignore the item.

```
Tokenizer:addItem(item: any)
```

#### Parameters:

* item: The item to be tokenized.

### addAllItems()

Tokenize all items in the item array. If an item already exists, then it will ignore that item.

```
Tokenizer:addAllItems(itemArray: [])
```

#### Parameters:

* itemArray: An array of items to be tokenized.

### convertTokenToItem()

Gets the item from a given token number. If no item is found from the given token number, then it will return nil.

```
Tokenizer:convertTokenToItem(tokenNumber: integer): any
```

#### Parameters:

* tokenNumber: A positive integer that represents the item.

#### Returns:

* item: An item that is retrieved by the token number.

### convertItemToToken()

Gets the token number from a given item. If no item is found, then it will return nil.

```
Tokenizer:convertItemToToken(item: any): integer
```

#### Parameters:

* item: The item to be converted into token.

#### Returns:

* token: A positive integer that represents the item.

### getTokenizedItemArray()

Gets an array of tokenized items from the tokenizer object.

```
Tokenizer:getTokenizedItemArray(): []
```

#### Returns:

* tokenizedItemArray: An array containing all the items. The position of the items in the array indicates the token number.

### setTokenizedItemArray()

Sets an array of tokenized items to the tokenizer object.

```
Tokenizer:setTokenizedItemArray(tokenizedItemArray: [])
```

#### Parameters:

* tokenizedItemArray: An array containing all the items. The position of the items in the array indicates the token number.




