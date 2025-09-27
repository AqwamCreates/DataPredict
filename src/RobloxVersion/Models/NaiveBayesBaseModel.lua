--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local BaseModel = require(script.Parent.BaseModel)

NaiveBayesBaseModel = {}

NaiveBayesBaseModel.__index = NaiveBayesBaseModel

setmetatable(NaiveBayesBaseModel, BaseModel)

function NaiveBayesBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseModel = BaseModel.new(parameterDictionary)

	setmetatable(NewBaseModel, NaiveBayesBaseModel)

	NewBaseModel:setName("NaiveBayesBaseModel")

	NewBaseModel:setClassName("NaiveBayesModel")

	return NewBaseModel
	
end

function NaiveBayesBaseModel:train(featureMatrix, labelVector)
	
	return self.trainFunction(featureMatrix, labelVector)
	
end


function NaiveBayesBaseModel:setTrainFunction(trainFunction)
	
	self.trainFunction = trainFunction
	
end

function NaiveBayesBaseModel:predict(featureMatrix, returnOriginalOutput)

	return self.predictFunction(featureMatrix, returnOriginalOutput)

end


function NaiveBayesBaseModel:setPredictFunction(predictFunction)

	self.predictFunction = predictFunction

end

function NaiveBayesBaseModel:generate(labelVector)
	
	return self.generateFunction(labelVector)
	
end

function NaiveBayesBaseModel:setGenerateFunction(generateFunction)
	
	self.generateFunction = generateFunction
	
end

return NaiveBayesBaseModel
