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

Regularizer = {}

Regularizer.__index = Regularizer

local defaultRegularizationMode = "L2"

local defaultLambda = 0.01

local function getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean
	
end

local function makeLambdaAtBiasZero(regularizationDerivatives)
	
	for i = 1, #regularizationDerivatives[1], 1 do
		
		regularizationDerivatives[1][i] = 0
		
	end
	
	return regularizationDerivatives
	
end

function Regularizer.new(lambda, regularizationMode, hasBias)
	
	local NewRegularizer = {}
	
	setmetatable(NewRegularizer, Regularizer)
	
	NewRegularizer.lambda = lambda or defaultLambda
	
	NewRegularizer.regularizationMode = regularizationMode or defaultRegularizationMode
	
	NewRegularizer.hasBias = getBooleanOrDefaultOption(hasBias, false)
	
	return NewRegularizer
	
end

function Regularizer:setParameters(lambda, regularizationMode, hasBias)
	
	self.lambda = lambda or self.lambda
	
	self.regularizationMode = regularizationMode or self.regularizationMode
	
	self.hasBias = getBooleanOrDefaultOption(hasBias, self.hasBias)
	
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
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		regularizationDerivatives = AqwamMatrixLibrary:multiply(ModelParametersSign, lambda, ModelParameters)
	
	elseif (regularizationMode == "L2") or (regularizationMode == "Ridge") then
		
		regularizationDerivatives = AqwamMatrixLibrary:multiply((2 * lambda), ModelParameters)
		
	elseif (regularizationMode == "L1+L2") or (regularizationMode == "ElasticNet") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		local regularizationDerivativesPart1 = AqwamMatrixLibrary:multiply(lambda, ModelParametersSign)
		
		local regularizationDerivativesPart2 = AqwamMatrixLibrary:multiply((2 * lambda), ModelParameters)
		
		regularizationDerivatives = AqwamMatrixLibrary:add(regularizationDerivativesPart1, regularizationDerivativesPart2)

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
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		if (self.hasBias) then AbsoluteModelParameters = makeLambdaAtBiasZero(AbsoluteModelParameters) end
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		regularizationValue = lambda * SumAbsoluteModelParameters
		
	elseif (regularizationMode == "L2") or (regularizationMode == "Ridge") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		regularizationValue = lambda * SumSquaredModelParameters
		
	elseif (regularizationMode == "L1+L2") or (regularizationMode == "ElasticNet") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
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