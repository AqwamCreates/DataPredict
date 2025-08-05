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
	
	NewLasso:setCalculateFunction(function(ModelParameters)
		
		local signMatrix = AqwamTensorLibrary:applyFunction(math.sign, ModelParameters)
		
		return AqwamTensorLibrary:multiply(signMatrix, NewLasso.lambda, ModelParameters)
		
	end)
	
	return NewLasso
	
end

return Lasso
