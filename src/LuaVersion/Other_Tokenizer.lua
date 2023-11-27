--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

Tokenizer = {}

Tokenizer.__index = Tokenizer

function Tokenizer.new(tokenizedItemArray)
	
	local NewTokenizer = {}
	
	setmetatable(NewTokenizer, Tokenizer)
		
	NewTokenizer.tokenizedItemArray = tokenizedItemArray or {}
	
	return NewTokenizer
	
end

function Tokenizer:addItem(item)
	
	if table.find(self.tokenizedItemArray, item) then return nil end

	table.insert(self.tokenizedItemArray, item)
	
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
