local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, ReinforcementLearningBaseModel)

function DeepClippedDoubleQLearningModel.new(discountFactor)

	local NewDeepClippedDoubleQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)

	NewDeepClippedDoubleQLearningModel.ModelParametersArray = {}
	
	NewDeepClippedDoubleQLearningModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepClippedDoubleQLearningModel.Model

		local maxQValues = {}

		for i = 1, 2, 1 do

			Model:setModelParameters(NewDeepClippedDoubleQLearningModel.ModelParametersArray[i], true)

			local predictedValue, maxQValue = Model:predict(currentFeatureVector)

			table.insert(maxQValues, maxQValue[1][1])

		end

		local maxQValue = math.min(table.unpack(maxQValues))

		local targetValue = rewardValue + (NewDeepClippedDoubleQLearningModel.discountFactor * maxQValue)
		
		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)
		
		local numberOfClasses = #ClassesList
		
		local temporalDifferenceVector = AqwamMatrixLibrary:createMatrix(1, 2)

		for i = 1, 2, 1 do

			Model:setModelParameters(NewDeepClippedDoubleQLearningModel.ModelParametersArray[i], true)

			local previousVector = Model:forwardPropagate(previousFeatureVector, true)

			local lastValue = previousVector[1][actionIndex]
			
			local temporalDifferenceError = targetValue - lastValue

			local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)
			
			lossVector[1][actionIndex] = temporalDifferenceError
			
			temporalDifferenceVector[1][i] = temporalDifferenceError
			
			Model:backPropagate(lossVector, true)

		end
		
		return temporalDifferenceVector

	end)

	return NewDeepClippedDoubleQLearningModel

end

function DeepClippedDoubleQLearningModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepClippedDoubleQLearningModel:setModelParameters1(ModelParameters1)

	self.ModelParametersArray[1] = ModelParameters1

end

function DeepClippedDoubleQLearningModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function DeepClippedDoubleQLearningModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function DeepClippedDoubleQLearningModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return DeepClippedDoubleQLearningModel
