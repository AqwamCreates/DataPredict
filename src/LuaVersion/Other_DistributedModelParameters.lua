--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local ModelParametersMerger = require("Other_ModelParametersMerger")

DistributedModelParameters = {}

DistributedModelParameters.__index = DistributedModelParameters

local defaultTotalNumberOfChildModelUpdatesToUpdateMainModel = 100

function DistributedModelParameters.new(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	local NewDistributedModelParameters = {}
	
	setmetatable(NewDistributedModelParameters, DistributedModelParameters)
	
	NewDistributedModelParameters.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or defaultTotalNumberOfChildModelUpdatesToUpdateMainModel
	
	NewDistributedModelParameters.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
	NewDistributedModelParameters.ModelParametersArray = {}
	
	NewDistributedModelParameters.isDistributedLearningRunning = false
	
	NewDistributedModelParameters.ModelParametersMerger = nil
	
	return NewDistributedModelParameters
	
end

function DistributedModelParameters:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	self.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or self.totalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedModelParameters:addModelParameters(ModelParameters)
	
	if not ModelParameters then error("No model parameters!") end
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel + 1

	table.insert(self.ModelParametersArray, ModelParameters)
	
end

function DistributedModelParameters:setModelParametersMerger(ModelParametersMerger)
	
	self.ModelParametersMerger = ModelParametersMerger or self.ModelParametersMerger
	
end

function DistributedModelParameters:setMainModelParameters(MainModelParameters)
	
	self.MainModelParameters = MainModelParameters
	
end

function DistributedModelParameters:getMainModelParameters()
	
	return self.MainModelParameters
	
end

function DistributedModelParameters:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()
	
	return self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedModelParameters:start()
	
	if (self.isDistributedLearningRunning == true) then error("The model is already running!") end
	
	self.isDistributedLearningRunning = true
	
	local modelParameterChangeCoroutine = coroutine.create(function()
		
		local ModelParametersArray = self.ModelParametersArray

		repeat
			
			task.wait()
			
			local totalNumberOfChildModelUpdatesToUpdateMainModel = self.totalNumberOfChildModelUpdatesToUpdateMainModel
			
			if (self.ModelParametersMerger == nil) then warn("No model parameters merger!") continue end
			
			if (self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel < totalNumberOfChildModelUpdatesToUpdateMainModel) then continue end
			
			self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
			
			local CurrentModelParametersArray = {}
			
			for i = 1, totalNumberOfChildModelUpdatesToUpdateMainModel, 1 do
				
				table.insert(CurrentModelParametersArray, ModelParametersArray[i])
				
				table.remove(ModelParametersArray, i)
				
			end
			
			self.MainModelParameters = self.ModelParametersMerger:merge(CurrentModelParametersArray)

		until (self.isDistributedLearningRunning == false)

	end)

	coroutine.resume(modelParameterChangeCoroutine)

	return modelParameterChangeCoroutine
		
end

function DistributedModelParameters:stop()
	
	self.isDistributedLearningRunning = false
	
end

function DistributedModelParameters:reset()
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
end

function DistributedModelParameters:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DistributedModelParameters