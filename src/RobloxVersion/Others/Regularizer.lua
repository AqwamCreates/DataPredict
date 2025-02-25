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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

Regularizer = {}

Regularizer.__index = Regularizer

setmetatable(Regularizer, BaseInstance)

local defaultRegularizationMode = "L2"

local defaultLambda = 0.01

local function makeLambdaAtBiasZero(regularizationDerivatives)
	
	for i = 1, #regularizationDerivatives[1], 1 do
		
		regularizationDerivatives[1][i] = 0
		
	end
	
	return regularizationDerivatives
	
end

function Regularizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewRegularizer = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewRegularizer, Regularizer)
	
	NewRegularizer:setName("Regularizer")
	
	NewRegularizer:setClassName("Regularizer")
	
	NewRegularizer.lambda = parameterDictionary.lambda or defaultLambda
	
	NewRegularizer.regularizationMode = parameterDictionary.regularizationMode or defaultRegularizationMode
	
	NewRegularizer.hasBias = NewRegularizer:getValueOrDefaultValue(parameterDictionary.hasBias, false)
	
	return NewRegularizer
	
end

function Regularizer:getLambda()
	
	return self.lambda
	
end

function Regularizer:calculateRegularizationDerivatives(ModelParameters)
	
	local ModelParametersSign
	
	local regularizationDerivatives

	local lambda =  self.lambda
	
	local regularizationMode = self.regularizationMode
	
	if (regularizationMode == "L1") or (regularizationMode == "Lasso") then
		
		ModelParametersSign = AqwamTensorLibrary:applyFunction(math.sign, ModelParameters)
		
		regularizationDerivatives = AqwamTensorLibrary:multiply(ModelParametersSign, lambda, ModelParameters)
	
	elseif (regularizationMode == "L2") or (regularizationMode == "Ridge") then
		
		regularizationDerivatives = AqwamTensorLibrary:multiply((2 * lambda), ModelParameters)
		
	elseif (regularizationMode == "L1+L2") or (regularizationMode == "ElasticNet") then
		
		ModelParametersSign = AqwamTensorLibrary:applyFunction(math.sign, ModelParameters)
		
		local regularizationDerivativesPart1 = AqwamTensorLibrary:multiply(lambda, ModelParametersSign)
		
		local regularizationDerivativesPart2 = AqwamTensorLibrary:multiply((2 * lambda), ModelParameters)
		
		regularizationDerivatives = AqwamTensorLibrary:add(regularizationDerivativesPart1, regularizationDerivativesPart2)

	else

		error("Regularization mode does not exist!")

	end
	
	if (self.hasBias) then regularizationDerivatives = makeLambdaAtBiasZero(regularizationDerivatives) end
	
	return regularizationDerivatives
	
end

function Regularizer:calculateRegularization(ModelParameters)
	
	local SquaredModelParameters 
	
	local AbsoluteModelParameters
	
	local SumSquaredModelParameters
	
	local SumAbsoluteModelParameters
	
	local regularizationValue
	
	local lambda =  self.lambda

	local regularizationMode = self.regularizationMode
	
	if (regularizationMode == "L1") or (regularizationMode == "Lasso") then
		
		AbsoluteModelParameters = AqwamTensorLibrary:applyFunction(math.abs, ModelParameters)
		
		if (self.hasBias) then AbsoluteModelParameters = makeLambdaAtBiasZero(AbsoluteModelParameters) end
		
		SumAbsoluteModelParameters = AqwamTensorLibrary:sum(AbsoluteModelParameters)
		
		regularizationValue = lambda * SumAbsoluteModelParameters
		
	elseif (regularizationMode == "L2") or (regularizationMode == "Ridge") then
		
		SquaredModelParameters = AqwamTensorLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamTensorLibrary:sum(SquaredModelParameters)
		
		regularizationValue = lambda * SumSquaredModelParameters
		
	elseif (regularizationMode == "L1+L2") or (regularizationMode == "ElasticNet") then
		
		SquaredModelParameters = AqwamTensorLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamTensorLibrary:sum(SquaredModelParameters)
		
		AbsoluteModelParameters = AqwamTensorLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamTensorLibrary:sum(AbsoluteModelParameters)
		
		local regularizationValuePart1 = lambda * SumSquaredModelParameters
		
		local regularizationValuePart2 = lambda * SumAbsoluteModelParameters
		
		regularizationValue = regularizationValuePart1 + regularizationValuePart2
		
	else
		
		error("Regularization mode does not exist!")
		
	end
	
	regularizationValue = regularizationValue / 2
	
	return regularizationValue
	
end

return Regularizer