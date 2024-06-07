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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningBaseModel = {}

ReinforcementLearningBaseModel.__index = ReinforcementLearningBaseModel

local defaultDiscountFactor = 0.95

function ReinforcementLearningBaseModel.new(discountFactor)
	
	local NewReinforcementLearningBaseModel = {}
	
	setmetatable(NewReinforcementLearningBaseModel, ReinforcementLearningBaseModel)
	
	NewReinforcementLearningBaseModel.discountFactor = discountFactor or defaultDiscountFactor
	
	return NewReinforcementLearningBaseModel
	
end

function ReinforcementLearningBaseModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
end

function ReinforcementLearningBaseModel:setModel(Model)
	
	self.Model = Model
	
end

function ReinforcementLearningBaseModel:getModel()

	return self.Model

end

function ReinforcementLearningBaseModel:setUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.Model:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	return self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

end

function ReinforcementLearningBaseModel:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction

	if not episodeUpdateFunction then return end

	episodeUpdateFunction()

end

function ReinforcementLearningBaseModel:extendResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningBaseModel:reset()

	if (self.resetFunction) then self.resetFunction() end

end

function ReinforcementLearningBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningBaseModel
