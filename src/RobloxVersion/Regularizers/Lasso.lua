--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseRegularizer = require(script.Parent.BaseRegularizer)

Lasso = {}

Lasso.__index = Lasso

setmetatable(Lasso, BaseRegularizer)

function Lasso.new(parameterDictionary)
	
	local NewLasso = BaseRegularizer.new(parameterDictionary)
	
	setmetatable(NewLasso, Lasso)
	
	NewLasso:setName("Lasso")
	
	NewLasso:setCalculateCostFunction(function(weightMatrix)
		
		weightMatrix = NewLasso:adjustWeightMatrix(weightMatrix)

		local absoluteWeightMatrix = AqwamTensorLibrary:applyFunction(math.abs, weightMatrix)

		local sumAbsoluteWeightMatrix = AqwamTensorLibrary:sum(absoluteWeightMatrix)

		return (NewLasso.lambda * sumAbsoluteWeightMatrix)

	end)
	
	NewLasso:setCalculateFunction(function(weightMatrix)
		
		weightMatrix = NewLasso:adjustWeightMatrix(weightMatrix)
		
		local signMatrix = AqwamTensorLibrary:applyFunction(math.sign, weightMatrix)
		
		return AqwamTensorLibrary:multiply(signMatrix, NewLasso.lambda, weightMatrix)
		
	end)
	
	return NewLasso
	
end

return Lasso
