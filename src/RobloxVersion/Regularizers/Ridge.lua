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

Ridge = {}

Ridge.__index = Ridge

setmetatable(Ridge, BaseRegularizer)

function Ridge.new(parameterDictionary)
	
	local NewRidge = BaseRegularizer.new(parameterDictionary)
	
	setmetatable(NewRidge, Ridge)
	
	NewRidge:setName("Ridge")
	
	NewRidge:setCalculateCostFunction(function(ModelParameters)
		
		ModelParameters = NewRidge:adjustModelParameters(ModelParameters)

		local squaredModelParameters = AqwamTensorLibrary:power(ModelParameters, 2)

		local sumSquaredModelParameters = AqwamTensorLibrary:sum(squaredModelParameters)

		return NewRidge.lambda * sumSquaredModelParameters

	end)
	
	NewRidge:setCalculateFunction(function(ModelParameters)
		
		ModelParameters = NewRidge:adjustModelParameters(ModelParameters)
		
		return AqwamTensorLibrary:multiply(2, NewRidge.lambda, ModelParameters)
		
	end)
	
	return NewRidge
	
end

return Ridge
