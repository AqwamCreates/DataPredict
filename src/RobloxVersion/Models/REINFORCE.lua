local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

REINFORCEModel = {}

REINFORCEModel.__index = REINFORCEModel

setmetatable(REINFORCEModel, ReinforcementLearningBaseModel)

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function REINFORCEModel.new(discountFactor)

	local NewREINFORCEModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewREINFORCEModel, REINFORCEModel)
	
	local targetVectorArray = {}
	
	local rewardArray = {}
	
	NewREINFORCEModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local predictedVector = NewREINFORCEModel.Model:predict(previousFeatureVector, true)
		
		local logPredictedVector = AqwamMatrixLibrary:applyFunction(math.log, predictedVector)
		
		local targetVector = AqwamMatrixLibrary:multiply(logPredictedVector, rewardValue)

		table.insert(targetVectorArray, targetVector)
		
		table.insert(rewardArray, rewardValue)

	end)
	
	NewREINFORCEModel:setEpisodeUpdateFunction(function()
		
		local Model = NewREINFORCEModel.Model
		
		local rewardsToGoArray = calculateRewardsToGo(rewardArray, NewREINFORCEModel.discountFactor)
		
		local ClassesList = Model:getClassesList()
		
		local lossVector = AqwamMatrixLibrary:createMatrix(1, #ClassesList)
		
		for i = 1, #targetVectorArray, 1 do
			
			local discountedReward = AqwamMatrixLibrary:multiply(targetVectorArray[i], rewardsToGoArray[i])
			
			lossVector = AqwamMatrixLibrary:add(lossVector, discountedReward)
			
		end
		
		local numberOfNeurons = Model:getTotalNumberOfNeurons(1)

		local inputVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeurons, 1)
		
		lossVector = AqwamMatrixLibrary:multiply(-1, lossVector)
		
		Model:forwardPropagate(inputVector, true)

		Model:backPropagate(lossVector, true)
		
		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)
	
	NewREINFORCEModel:extendResetFunction(function()

		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)

	return NewREINFORCEModel

end

function REINFORCEModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return REINFORCEModel
