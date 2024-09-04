--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

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

function ReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function ReinforcementLearningBaseModel:setCategoricalEpisodeUpdateFunction(categoricalEpisodeUpdateFunction)

	self.categoricalEpisodeUpdateFunction = categoricalEpisodeUpdateFunction

end

function ReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
	
	self.diagonalGaussianUpdateFunction = diagonalGaussianUpdateFunction
	
end

function ReinforcementLearningBaseModel:setDiagonalGaussianEpisodeUpdateFunction(diagonalGaussianEpisodeUpdateFunction)
	
	self.diagonalGaussianEpisodeUpdateFunction = diagonalGaussianEpisodeUpdateFunction
	
end

function ReinforcementLearningBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.Model:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local categoricalUpdateFunction = self.categoricalUpdateFunction
	
	if (categoricalUpdateFunction) then
		
		return categoricalUpdateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
	else
		
		error("Categorical update function is not implemented!")
		
	end

end

function ReinforcementLearningBaseModel:categoricalEpisodeUpdate()

	local categoricalEpisodeUpdateFunction = self.categoricalEpisodeUpdateFunction
	
	if (categoricalEpisodeUpdateFunction) then
		
		return categoricalEpisodeUpdateFunction()
		
	else
		
		error("Categorical episode update function is not implemented!")
		
	end

end

function ReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)
	
	local diagonalGaussianUpdateFunction = self.diagonalGaussianUpdateFunction
	
	if (diagonalGaussianUpdateFunction) then
		
		return diagonalGaussianUpdateFunction(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)
		
	else
		
		error("Diagonal Gaussian update function is not implemented!")
		
	end
	
end

function ReinforcementLearningBaseModel:diagonalGaussianEpisodeUpdate()

	local diagonalGaussianEpisodeUpdateFunction = self.diagonalGaussianEpisodeUpdateFunction
	
	if (diagonalGaussianEpisodeUpdateFunction) then
		
		return diagonalGaussianEpisodeUpdateFunction()
		
	else
		
		error("Diagonal Gaussian episode update function is not implemented!")
		
	end

end

function ReinforcementLearningBaseModel:setCategoricalResetFunction(categoricalResetFunction)

	self.categoricalResetFunction = categoricalResetFunction

end

function ReinforcementLearningBaseModel:setDiagonalGaussianResetFunction(diagonalGaussianResetFunction)

	self.diagonalGaussianResetFunction = diagonalGaussianResetFunction

end

function ReinforcementLearningBaseModel:categoricalReset()
	
	local categoricalResetFunction = self.categoricalResetFunction

	if (categoricalResetFunction) then 
		
		return categoricalResetFunction() 
		
	else
		
		error("Categorical reset function is not implemented!")
		
	end

end

function ReinforcementLearningBaseModel:diagonalGaussianReset()
	
	local diagonalGaussianResetFunction = self.diagonalGaussianResetFunction

	if (diagonalGaussianResetFunction) then 
		
		return diagonalGaussianResetFunction()
		
	else
		
		error("Diagonal Gaussian reset function is not implemented!")
		
	end

end

function ReinforcementLearningBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningBaseModel