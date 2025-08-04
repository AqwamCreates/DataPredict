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

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

DeepMonteCarloControlModel = {}

DeepMonteCarloControlModel.__index = DeepMonteCarloControlModel

setmetatable(DeepMonteCarloControlModel, DeepReinforcementLearningBaseModel)

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function DeepMonteCarloControlModel.new(parameterDictionary)

	local NewDeepMonteCarloControlModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepMonteCarloControlModel, DeepMonteCarloControlModel)
	
	NewDeepMonteCarloControlModel:setName("DeepMonteCarloControl")
	
	local featureVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewDeepMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		table.insert(featureVectorHistory, previousFeatureVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewDeepMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		local Model = NewDeepMonteCarloControlModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewDeepMonteCarloControlModel.discountFactor)
		
		for h, featureVector in ipairs(featureVectorHistory) do
			
			local averageRewardToGo = rewardToGoArray[h] / h
			
			Model:forwardPropagate(featureVector, true)

			Model:update(averageRewardToGo, true)
			
		end
		
		table.clear(featureVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewDeepMonteCarloControlModel:setResetFunction(function()
		
		table.clear(featureVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewDeepMonteCarloControlModel

end

return DeepMonteCarloControlModel
