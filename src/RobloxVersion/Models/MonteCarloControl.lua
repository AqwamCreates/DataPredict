--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

MonteCarloControlModel = {}

MonteCarloControlModel.__index = MonteCarloControlModel

setmetatable(MonteCarloControlModel, ReinforcementLearningBaseModel)

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function MonteCarloControlModel.new(parameterDictionary)

	local NewMonteCarloControlModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewMonteCarloControlModel, MonteCarloControlModel)
	
	NewMonteCarloControlModel:setName("MonteCarloControl")
	
	local featureVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		table.insert(featureVectorHistory, previousFeatureVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local Model = NewMonteCarloControlModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewMonteCarloControlModel.discountFactor)
		
		for h, featureVector in ipairs(featureVectorHistory) do
			
			local averageRewardToGo = rewardToGoArray[h] / h
			
			Model:forwardPropagate(featureVector, true, true)

			Model:backwardPropagate(averageRewardToGo, true)
			
		end
		
		table.clear(featureVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewMonteCarloControlModel:setResetFunction(function()
		
		table.clear(featureVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewMonteCarloControlModel

end

return MonteCarloControlModel