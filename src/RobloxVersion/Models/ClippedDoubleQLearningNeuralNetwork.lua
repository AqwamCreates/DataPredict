local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

ClippedDoubleQLearningNeuralNetworkModel = {}

ClippedDoubleQLearningNeuralNetworkModel.__index = ClippedDoubleQLearningNeuralNetworkModel

setmetatable(ClippedDoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function ClippedDoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, discountFactor)

	local NewClippedDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, discountFactor)

	NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}
	
	NewClippedDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local maxQValues = {}

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i], true)

			local predictedValue, maxQValue = NewClippedDoubleQLearningNeuralNetworkModel:predict(currentFeatureVector)

			table.insert(maxQValues, maxQValue[1][1])

		end

		local maxQValue = math.min(table.unpack(maxQValues))

		local targetValue = rewardValue + (NewClippedDoubleQLearningNeuralNetworkModel.discountFactor * maxQValue)

		local actionIndex = table.find(NewClippedDoubleQLearningNeuralNetworkModel.ClassesList, action)
		
		local numberOfClasses = #NewClippedDoubleQLearningNeuralNetworkModel:getClassesList()
		
		local temporalDifferenceVector = AqwamMatrixLibrary:createMatrix(1, 2)

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i], true)

			local previousVector = NewClippedDoubleQLearningNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)

			local lastValue = previousVector[1][actionIndex]
			
			local temporalDifferenceError = targetValue - lastValue

			local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)
			
			lossVector[1][actionIndex] = temporalDifferenceError
			
			temporalDifferenceVector[1][i] = temporalDifferenceError

			NewClippedDoubleQLearningNeuralNetworkModel:train(previousFeatureVector, lossVector)

		end
		
		return temporalDifferenceVector

	end)

	return NewClippedDoubleQLearningNeuralNetworkModel

end

function ClippedDoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.discountFactor =  discountFactor or self.discountFactor

end

function ClippedDoubleQLearningNeuralNetworkModel:setModelParameters1(ModelParameters1)

	self.ModelParametersArray[1] = ModelParameters1

end

function ClippedDoubleQLearningNeuralNetworkModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function ClippedDoubleQLearningNeuralNetworkModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function ClippedDoubleQLearningNeuralNetworkModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return ClippedDoubleQLearningNeuralNetworkModel
