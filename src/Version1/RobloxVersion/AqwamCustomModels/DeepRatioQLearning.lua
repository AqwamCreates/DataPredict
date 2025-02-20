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

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.Parent.Models.ReinforcementLearningBaseModel)

DeepRatioQLearningModel = {}

DeepRatioQLearningModel.__index = DeepRatioQLearningModel

setmetatable(DeepRatioQLearningModel, ReinforcementLearningBaseModel)

function DeepRatioQLearningModel.new(discountFactor)

	local NewDeepRatioQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewDeepRatioQLearningModel, DeepRatioQLearningModel)
	
	NewDeepRatioQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepRatioQLearningModel.Model
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local currentQVector = NewDeepRatioQLearningModel:predict(currentFeatureVector, true)
		
		local previousQVector = NewDeepRatioQLearningModel:predict(previousFeatureVector, true)
		
		local currentMaxQValue = AqwamMatrixLibrary:findMaximumValue(currentQVector)
		
		local previousMaximumQValue, previousActionIndex = AqwamMatrixLibrary:findMaximumValue(previousQVector)
		
		local qRatioVector = AqwamMatrixLibrary:divide(currentQVector, previousQVector)
		
		local maxQRatio = AqwamMatrixLibrary:findMaximumValue(qRatioVector)
		
		local newQValue = maxQRatio * previousMaximumQValue
		
		local actionIndex = table.find(ClassesList, action)
		
		local targetValue = rewardValue + (NewDeepRatioQLearningModel.discountFactor * newQValue)

		local lastValue = previousQVector[1][previousActionIndex[2]]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		lossVector[1][actionIndex] = temporalDifferenceError
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:backwardPropagate(lossVector, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepRatioQLearningModel:setEpisodeUpdateFunction(function() end)
	
	NewDeepRatioQLearningModel:setResetFunction(function() end)

	return NewDeepRatioQLearningModel

end

function DeepRatioQLearningModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

return DeepRatioQLearningModel