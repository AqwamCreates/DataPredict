local ModelParametersMerger = require(script.Parent.Parent.Others.ModelParametersMerger)

DistributedTraining = {}

DistributedTraining.__index = DistributedTraining

local defaultTotalNumberOfChildModelUpdatesToUpdateMainModel = 100

function DistributedTraining.new(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	local NewDistributedTraining = {}
	
	setmetatable(NewDistributedTraining, DistributedTraining)
	
	NewDistributedTraining.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or defaultTotalNumberOfChildModelUpdatesToUpdateMainModel
	
	NewDistributedTraining.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
	NewDistributedTraining.ModelArray = {}
	
	NewDistributedTraining.isDistributedLearningRunning = false
	
	NewDistributedTraining.ModelParametersMerger = ModelParametersMerger.new(nil, nil, "Average")
	
	return NewDistributedTraining
	
end

function DistributedTraining:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	self.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or self.totalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedTraining:addModel(Model)
	
	if not Model then error("Model is empty!") end

	table.insert(self.ModelArray, Model)
	
end

function DistributedTraining:train(featureVector, labelVector, modelNumber)

	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel += 1

	local Model = self.ModelArray[modelNumber]

	if not Model then error("No model!") end

	return Model:train(featureVector, labelVector)

end

function DistributedTraining:predict(featureVector, returnOriginalOutput, modelNumber)

	local Model = self.ModelArray[modelNumber]

	if not Model then error("No model!") end

	return Model:predict(featureVector, returnOriginalOutput)

end

function DistributedTraining:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, modelNumber)
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel += 1
	
	local Model = self.ModelArray[modelNumber]
	
	if not Model then error("No model!") end
	
	return Model:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
end

function DistributedTraining:setMainModelParameters(MainModelParameters)
	
	self.MainModelParameters = MainModelParameters
	
end

function DistributedTraining:getMainModelParameters()
	
	return self.MainModelParameters
	
end

function DistributedTraining:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()
	
	return self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedTraining:start()
	
	if (self.isDistributedLearningRunning == true) then error("The model is already running!") end
	
	self.isDistributedLearningRunning = true
	
	local trainCoroutine = coroutine.create(function()

		repeat
			
			task.wait()
			
			if (self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel < self.totalNumberOfChildModelUpdatesToUpdateMainModel) then continue end
			
			self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
			
			local ModelParametersArray = {}
			
			for _, Model in ipairs(self.ModelArray) do table.insert(ModelParametersArray, Model:getModelParameters()) end
			
			self.ModelParametersMerger:setModelParameters(table.unpack(ModelParametersArray))
			
			local MainModelParameters = self.ModelParametersMerger:generate()
			
			for _, Model in ipairs(self.ModelArray) do Model:setModelParameters(MainModelParameters) end
			
			self.MainModelParameters = MainModelParameters

		until (self.isDistributedLearningRunning == false)

	end)

	coroutine.resume(trainCoroutine)

	return trainCoroutine
		
end

function DistributedTraining:stop()
	
	self.isDistributedLearningRunning = false
	
end

function DistributedTraining:reset()
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
end

function DistributedTraining:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DistributedTraining
