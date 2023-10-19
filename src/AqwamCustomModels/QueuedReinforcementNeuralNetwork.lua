local NeuralNetworkModel = require(script.Parent.Parent.Models.NeuralNetwork)

QueuedReinforcementNeuralNetworkModel = {}

QueuedReinforcementNeuralNetworkModel.__index = QueuedReinforcementNeuralNetworkModel

setmetatable(QueuedReinforcementNeuralNetworkModel, NeuralNetworkModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

function QueuedReinforcementNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)
	
	local NewQueuedReinforcementNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost)
	
	setmetatable(NewQueuedReinforcementNeuralNetworkModel, QueuedReinforcementNeuralNetworkModel)
	
	return NewQueuedReinforcementNeuralNetworkModel
	
end

function QueuedReinforcementNeuralNetworkModel:checkIfRewardAndPunishValuesAreGiven(rewardValue, punishValue)

	if (rewardValue == nil) then error("Reward value is nil!") end

	if (punishValue == nil) then error("Punish value is nil!") end

	if (rewardValue < 0) then error("Reward value must be a positive integer!") end

	if (punishValue < 0) then error("Punish value must be a positive integer!") end

end

function QueuedReinforcementNeuralNetworkModel:start(rewardValue, punishValue, showPredictedLabel, showIdleWarning, showWaitingForLabelWarning)

	if (self.IsQueuedReinforcementRunning == true) then error("Queued reinforcement is already active!") end
	
	if (#self.ClassesList == 0) then error("Classes list is not set!") end
	
	if (self.ModelParameters == nil) then self:generateLayers() end

	self:checkIfRewardAndPunishValuesAreGiven(rewardValue, punishValue)

	self.FeatureVectorQueue = {}

	self.LabelQueue = {}

	self.PredictedLabelQueue = {}

	self.ForwardPropagationTableQueue = {}

	self.ZTableQueue = {}
	
	self.CostArrayQueue = {}

	self.IsQueuedReinforcementRunning = true

	if (showPredictedLabel == nil) then showPredictedLabel = false else showPredictedLabel = showPredictedLabel end

	if (showIdleWarning == nil) then showIdleWarning = true else showIdleWarning = showIdleWarning end

	if (showWaitingForLabelWarning == nil) then showWaitingForLabelWarning = false else showWaitingForLabelWarning = showWaitingForLabelWarning end

	local waitInterval = 0.1

	local idleDuration = 0

	local waitDuration = 0

	local idleWarningIssued = false

	local labelWarningIssued = false
	
	local isCurrentlyBackpropagating = false
	
	local infinityCostWarningIssued = false
	
	local PreviousModelParameters

	local predictCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			idleDuration += waitInterval

			if (idleDuration >= 30) and (idleWarningIssued == false) and (showIdleWarning == true) then 

				warn("The neural network has been idle for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				idleWarningIssued = true
				
				continue

			elseif (#self.FeatureVectorQueue == 0) then continue

			elseif (self.IsQueuedReinforcementRunning == false) then break end

			local forwardPropagateTable, zTable = self:forwardPropagate(self.FeatureVectorQueue[1], self.ModelParameters, self.activationFunction)

			table.insert(self.ForwardPropagationTableQueue, forwardPropagateTable)

			table.insert(self.ZTableQueue, zTable)

			local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]

			local predictedLabelVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

			table.insert(self.PredictedLabelQueue, predictedLabelVector[1][1])

			table.remove(self.FeatureVectorQueue, 1)

			idleDuration = 0

			idleWarningIssued = false

		until (self.IsQueuedReinforcementRunning == false)

	end)


	local reinforcementCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			waitDuration += waitInterval

			if (waitDuration >= 30) and (labelWarningIssued == false) and (showWaitingForLabelWarning == true) then

				warn("The neural network has been waiting for a label for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				labelWarningIssued = true
				
				continue

			elseif (#self.LabelQueue == 0) or (#self.PredictedLabelQueue == 0) or (#self.ForwardPropagationTableQueue == 0) or (#self.ZTableQueue == 0) or (isCurrentlyBackpropagating == true) then continue

			elseif (self.IsQueuedReinforcementRunning == false) then break end
			
			isCurrentlyBackpropagating = true

			if (showPredictedLabel == true) then print("Predicted Label: " .. self.PredictedLabelQueue[1] .. "\t\t\t\tActual Label: " .. self.LabelQueue[1]) end
			
			local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

			local logisticMatrix = self:convertLabelVectorToLogisticMatrix(self.LabelQueue[1])

			local forwardPropagationTable = self.ForwardPropagationTableQueue[1]

			local allOutputsMatrix = forwardPropagationTable[#forwardPropagationTable]
			
			local cost = self:calculateCost(allOutputsMatrix, logisticMatrix, 1)
			
			if (cost == math.huge) and (infinityCostWarningIssued == false) then 
				
				warn("The model diverged! Reverting to previous model parameters! Please repeat the experiment again or change the argument values if this warning occurs often.") 
				
				infinityCostWarningIssued = true
				
				self.ModelParameters = PreviousModelParameters
				
			end
			
			table.insert(self.CostArrayQueue, cost)
			
			PreviousModelParameters = self.ModelParameters

			local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

			local backwardPropagateTable = self:backPropagate(lossMatrix, self.ZTableQueue[1])

			local deltaTable = self:calculateDelta(self.ForwardPropagationTableQueue[1], backwardPropagateTable, 1)
			
			local multiplyFactor
			
			if (self.PredictedLabelQueue[1] == self.LabelQueue[1]) then

				multiplyFactor = rewardValue

			else

				multiplyFactor = punishValue
				
			end
			
			self.ModelParameters = self:gradientDescent(multiplyFactor, deltaTable, 1)
			
			isCurrentlyBackpropagating = false
			
			waitDuration = 0

			labelWarningIssued = false

			infinityCostWarningIssued = false

			table.remove(self.LabelQueue, 1)

			table.remove(self.ZTableQueue, 1)

			table.remove(self.ForwardPropagationTableQueue, 1)
			
			task.spawn(function()
				
				for frame = 1, 70 do task.wait() end -- to allow cost to be fetched. Otherwise it will remove it before it can be fetched!
				
				table.remove(self.PredictedLabelQueue, 1)
				
				table.remove(self.CostArrayQueue, 1)
				
			end)

		until (self.IsQueuedReinforcementRunning == false)	

	end)

	local resetCoroutine = coroutine.create(function()

		repeat task.wait(waitInterval) until (self.IsQueuedReinforcementRunning == false)

		self.IsQueuedReinforcementRunning = nil

		self.FeatureVectorQueue = nil

		self.LabelQueue = nil

		self.PredictedLabelQueue = nil

		self.ForwardPropagationTableQueue = nil

		self.ZTableQueue = nil

		waitInterval = nil

		idleDuration = nil

		waitDuration = nil

		idleWarningIssued = nil

		labelWarningIssued = nil
		
		isCurrentlyBackpropagating = nil
		
		infinityCostWarningIssued = nil
		
		PreviousModelParameters = nil

	end)

	coroutine.resume(predictCoroutine)

	coroutine.resume(reinforcementCoroutine)

	coroutine.resume(resetCoroutine)

	return predictCoroutine, reinforcementCoroutine, resetCoroutine

end

function QueuedReinforcementNeuralNetworkModel:stop()

	self.IsQueuedReinforcementRunning = false

end

function QueuedReinforcementNeuralNetworkModel:addFeatureVector(featureVector)

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	table.insert(self.FeatureVectorQueue, featureVector)

end

function QueuedReinforcementNeuralNetworkModel:addLabel(label)

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	table.insert(self.LabelQueue, label)

end

function QueuedReinforcementNeuralNetworkModel:returnPredictedLabel()

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	return self.PredictedLabelQueue[1]

end

function QueuedReinforcementNeuralNetworkModel:returnCost()

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	return self.CostArrayQueue[1]

end

return QueuedReinforcementNeuralNetworkModel
