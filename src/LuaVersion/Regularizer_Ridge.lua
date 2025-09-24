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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseRegularizer = require("Regularizer_BaseRegularizer")

Ridge = {}

Ridge.__index = Ridge

setmetatable(Ridge, BaseRegularizer)

function Ridge.new(parameterDictionary)
	
	local NewRidge = BaseRegularizer.new(parameterDictionary)
	
	setmetatable(NewRidge, Ridge)
	
	NewRidge:setName("Ridge")
	
	NewRidge:setCalculateCostFunction(function(ModelParameters)

		local squaredModelParameters = AqwamTensorLibrary:power(ModelParameters, 2)

		if (NewRidge.hasBias) then squaredModelParameters = NewRidge:makeLambdaAtBiasZero(squaredModelParameters) end

		local sumSquaredModelParameters = AqwamTensorLibrary:sum(squaredModelParameters)

		return NewRidge.lambda * sumSquaredModelParameters

	end)
	
	NewRidge:setCalculateFunction(function(ModelParameters)
		
		return AqwamTensorLibrary:multiply(2, NewRidge.lambda, ModelParameters)
		
	end)
	
	return NewRidge
	
end

return Ridge
