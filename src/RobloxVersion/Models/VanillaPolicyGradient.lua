local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningActorCriticNeuralNetworkBaseModel)

VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

setmetatable(VanillaPolicyGradientModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)

function VanillaPolicyGradientModel.new(discountFactor)
	
	local NewVanillaPolicyGradientModel = ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(discountFactor)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)

	local advantageHistory = {}

	local gradientHistory = {}

	NewVanillaPolicyGradientModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local allOutputsMatrix = NewVanillaPolicyGradientModel.ActorModel:predict(previousFeatureVector, true)

		local logOutputMatrix = AqwamMatrixLibrary:applyFunction(math.log, allOutputsMatrix)
		
		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local gradientMatrix = AqwamMatrixLibrary:multiply(logOutputMatrix, advantageValue)
		
		table.insert(advantageHistory, advantageValue)

		table.insert(gradientHistory, gradientMatrix[1])
		
		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function()

		local sumGradient = AqwamMatrixLibrary:verticalSum(gradientHistory)

		local sumAdvantage = 0
		
		for _, advantageValue in ipairs(advantageHistory) do
			
			sumAdvantage += advantageValue
			
		end
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures, hasBias = ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(-sumGradient, true)
		CriticModel:backPropagate(sumAdvantage, true)
		
		table.clear(advantageHistory)

		table.clear(gradientHistory)

	end)

	NewVanillaPolicyGradientModel:extendResetFunction(function()

		table.clear(advantageHistory)

		table.clear(gradientHistory)

	end)
	
	return NewVanillaPolicyGradientModel
	
end

return VanillaPolicyGradientModel
