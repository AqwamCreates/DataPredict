local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ReinforcingNeuralNetworkModel = {}

ReinforcingNeuralNetworkModel.__index = ReinforcingNeuralNetworkModel

setmetatable(ReinforcingNeuralNetworkModel, NeuralNetworkModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

function ReinforcingNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	local NewReinforcingNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	setmetatable(NewReinforcingNeuralNetworkModel, ReinforcingNeuralNetworkModel)

	return NewReinforcingNeuralNetworkModel

end

function ReinforcingNeuralNetworkModel:checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)

	if (rewardValue == nil) then error("Reward value is nil!") end

	if (punishValue == nil) then error("Punish value is nil!") end

	if (rewardValue < 0) then error("Reward value must be a positive integer!") end

	if (punishValue < 0) then error("Punish value must be a positive integer!") end

end

function ReinforcingNeuralNetworkModel:reset()
	
	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end
	
end

function ReinforcingNeuralNetworkModel:reinforce(featureVector, label, rewardValue, punishValue)
	
	if (self.ModelParameters == nil) then self:generateLayers() end

	self:checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)
	
	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(label)

	local forwardPropagateTable, zTable = self:forwardPropagate(featureVector)

	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]

	local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local backwardPropagateTable = self:backPropagate(lossMatrix, zTable)

	local deltaTable = self:calculateDelta(forwardPropagateTable, backwardPropagateTable)

	local predictedLabel, probability = self:getLabelFromOutputVector(allOutputsMatrix)

	local multiplyFactor

	if (predictedLabel == label) then

		multiplyFactor = rewardValue

	else

		multiplyFactor = punishValue

	end

	self.ModelParameters = self:gradientDescent(multiplyFactor, deltaTable, 1)

	return predictedLabel, probability

end

return ReinforcingNeuralNetworkModel
