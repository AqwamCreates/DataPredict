local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

REINFORCENeuralNetworkModel = {}

REINFORCENeuralNetworkModel.__index = REINFORCENeuralNetworkModel

setmetatable(REINFORCENeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward += rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function REINFORCENeuralNetworkModel.new(maxNumberOfIterations, discountFactor)

	local NewREINFORCENeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, discountFactor)
	
	setmetatable(NewREINFORCENeuralNetworkModel, REINFORCENeuralNetworkModel)
	
	local targetVectorArray = {}
	
	local rewardArray = {}
	
	NewREINFORCENeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local predictedVector = NewREINFORCENeuralNetworkModel:predict(previousFeatureVector, true)
		
		local logPredictedVector = AqwamMatrixLibrary:applyFunction(math.log, predictedVector)
		
		local targetVector = AqwamMatrixLibrary:multiply(logPredictedVector, rewardValue)

		table.insert(targetVectorArray, targetVector)
		
		table.insert(rewardArray, rewardValue)

	end)
	
	NewREINFORCENeuralNetworkModel:setEpisodeUpdateFunction(function()
		
		local rewardsToGoArray = calculateRewardsToGo(rewardArray, discountFactor)
		
		local lossVector = AqwamMatrixLibrary:createMatrix(1, #NewREINFORCENeuralNetworkModel.ClassesList)
		
		for i = 1, #targetVectorArray, 1 do
			
			local discountedReward = AqwamMatrixLibrary:multiply(targetVectorArray[i], rewardsToGoArray[i])
			
			lossVector = AqwamMatrixLibrary:add(lossVector, discountedReward)
			
		end
		
		local numberOfNeurons = NewREINFORCENeuralNetworkModel.numberOfNeuronsTable[1] + NewREINFORCENeuralNetworkModel.hasBiasNeuronTable[1]

		local inputVector = {table.create(numberOfNeurons, 1)}
		
		lossVector = AqwamMatrixLibrary:multiply(-1, lossVector)
		
		NewREINFORCENeuralNetworkModel:forwardPropagate(inputVector, true)

		NewREINFORCENeuralNetworkModel:backPropagate(lossVector, true)
		
		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)
	
	NewREINFORCENeuralNetworkModel:extendResetFunction(function()

		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)

	return NewREINFORCENeuralNetworkModel

end

function REINFORCENeuralNetworkModel:setParameters(maxNumberOfIterations, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.discountFactor = discountFactor or self.discountFactor

end

return REINFORCENeuralNetworkModel
