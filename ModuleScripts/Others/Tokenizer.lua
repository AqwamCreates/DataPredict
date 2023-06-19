Tokenizer = {}

Tokenizer.__index = Tokenizer

function Tokenizer.new(tokenizedItemArray)
	
	local NewTokenizer = {}
	
	setmetatable(NewTokenizer, Tokenizer)
		
	NewTokenizer.tokenizedItemArray = tokenizedItemArray or {}
	
	return NewTokenizer
	
end

function Tokenizer:addItem(item)
	
	if (table.find(self.tokenizedItemArray, item) == nil) then

		table.insert(self.tokenizedItemArray, item)

	end
	
end

function Tokenizer:addAllItems(itemArray)
	
	repeat
		
		self:addItem(itemArray[1])

		table.remove(itemArray, 1)

	until (#itemArray <= 0)
	
end

function Tokenizer:convertTokenToItem(tokenNumber)
	
	return self.tokenizedItemArray[tokenNumber]
	
end

function Tokenizer:convertItemToToken(item)

	return table.find(self.tokenizedItemArray, item)

end

function Tokenizer:getTokenizedItemArray()
	
	return self.tokenizedItemArray
	
end

function Tokenizer:setTokenizedItemArray(tokenizedItemArray)

	self.tokenizedItemArray = tokenizedItemArray

end

return Tokenizer
