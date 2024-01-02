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

local BaseOptimizer = require("Model_BaseOptimizer")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

MomentumOptimizer = {}

MomentumOptimizer.__index = MomentumOptimizer

setmetatable(MomentumOptimizer, BaseOptimizer)

local defaultDecayRate = 0.1

function MomentumOptimizer.new(decayRate)
	
	local NewMomentumOptimizer = BaseOptimizer.new("Momentum")
	
	setmetatable(NewMomentumOptimizer, MomentumOptimizer)
	
	NewMomentumOptimizer.decayRate = decayRate or defaultDecayRate
	
	NewMomentumOptimizer.velocity = nil
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setCalculationFunction(function(learningRate, costFunctionDerivatives)
		
		NewMomentumOptimizer.velocity = NewMomentumOptimizer.velocity or AqwamMatrixLibrary:createMatrix(#costFunctionDerivatives, #costFunctionDerivatives[1])

		local VelocityPart1 = AqwamMatrixLibrary:multiply(NewMomentumOptimizer.decayRate, NewMomentumOptimizer.velocity)

		local VelocityPart2 = AqwamMatrixLibrary:multiply(learningRate, costFunctionDerivatives)

		NewMomentumOptimizer.velocity = AqwamMatrixLibrary:add(VelocityPart1, VelocityPart2)

		costFunctionDerivatives = NewMomentumOptimizer.velocity

		return costFunctionDerivatives
		
	end)
	
	--------------------------------------------------------------------------------
	
	NewMomentumOptimizer:setResetFunction(function()
		
		NewMomentumOptimizer.velocity = nil
		
	end) 
	
	return NewMomentumOptimizer
	
end

function MomentumOptimizer:setDecayRate(decayRate)
	
	self.decayRate = decayRate
	
end

return MomentumOptimizer
