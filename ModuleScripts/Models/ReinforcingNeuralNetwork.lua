local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ReinforcingNeuralNetworkModel = {}

ReinforcingNeuralNetworkModel.__index = ReinforcingNeuralNetworkModel

setmetatable(ReinforcingNeuralNetworkModel, NeuralNetworkModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

function ReinforcingNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)
	
	maxNumberOfIterations = maxNumberOfIterations or 1

	local NewReinforcingNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)

	setmetatable(NewReinforcingNeuralNetworkModel, ReinforcingNeuralNetworkModel)

	return NewReinforcingNeuralNetworkModel

end

function ReinforcingNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

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

function ReinforcingNeuralNetworkModel:reinforce(featureVector, labelVector, rewardValue, punishValue, returnOriginalOutput)
	
	if (self.ModelParameters == nil) then self:generateLayers() end

	self:checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)
	
	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)
	
	if (#labelVector[1] == 1) then
		
		logisticMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)
		
	else

		logisticMatrix = labelVector
		
	end

	local forwardPropagateTable, zTable = self:forwardPropagate(featureVector)

	local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]

	local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

	local backwardPropagateTable = self:backPropagate(lossMatrix, zTable)

	local deltaTable = self:calculateDelta(forwardPropagateTable, backwardPropagateTable)

	local predictedVector, probabilityVector = self:getLabelFromOutputMatrix(allOutputsMatrix)
	
	local predictedLabel = predictedVector[1][1]
	
	local label = labelVector[1][1]
	
	local areLabelsEqual = (predictedLabel == label) 
	
	local probability = probabilityVector[1][1]

	local multiplyFactor = (areLabelsEqual and rewardValue) or punishValue

	self.ModelParameters = self:gradientDescent(multiplyFactor, deltaTable, self.maxNumberOfIterations)
	
	if (returnOriginalOutput == true) then return allOutputsMatrix end

	return predictedLabel, probability

end

return ReinforcingNeuralNetworkModel
