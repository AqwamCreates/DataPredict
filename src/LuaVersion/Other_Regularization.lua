--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]
local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

Regularization = {}

Regularization.__index = Regularization

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

function Regularization.new(lambda, regularizationMode, hasBias)
	
	local NewRegularization = {}
	
	setmetatable(NewRegularization, Regularization)
	
	NewRegularization.lambda = lambda or defaultLambda
	
	NewRegularization.regularizationMode = regularizationMode or defaultRegularizationMode
	
	NewRegularization.hasBias = getBooleanOrDefaultOption(hasBias, false)
	
	return NewRegularization
	
end

function Regularization:setParameters(lambda, regularizationMode, hasBias)
	
	self.lambda = lambda or self.lambda
	
	self.regularizationMode = regularizationMode or self.regularizationMode
	
	self.hasBias = getBooleanOrDefaultOption(hasBias, self.hasBias)
	
end

function Regularization:getLambda()
	
	return self.lambda
	
end

function Regularization:calculateRegularizationDerivatives(ModelParameters, numberOfData)
	
	local ModelParametersSign
	
	local regularizationDerivatives
	
	if (self.regularizationMode == "L1") or (self.regularizationMode == "Lasso") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		regularizationDerivatives = AqwamMatrixLibrary:multiply(ModelParametersSign, self.lambda, ModelParameters)
		
		if (self.hasBias) then regularizationDerivatives = makeLambdaAtBiasZero(regularizationDerivatives) end
	
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		regularizationDerivatives = AqwamMatrixLibrary:multiply(2, self.lambda, ModelParameters)
		
		if (self.hasBias) then regularizationDerivatives = makeLambdaAtBiasZero(regularizationDerivatives) end
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		local RegularizationDerivativePart1 = AqwamMatrixLibrary:multiply(self.lambda, ModelParametersSign)
		
		local RegularizationDerivativePart2 = AqwamMatrixLibrary:multiply(self.lambda, ModelParameters)
		
		regularizationDerivatives = AqwamMatrixLibrary:add(RegularizationDerivativePart1, RegularizationDerivativePart2)
		
		if (self.hasBias) then regularizationDerivatives = makeLambdaAtBiasZero(regularizationDerivatives) end

	else

		error("Regularization Mode Does Not Exist!")

	end
	
	regularizationDerivatives = AqwamMatrixLibrary:divide(regularizationDerivatives, numberOfData)
	
	return regularizationDerivatives
	
end

function Regularization:calculateRegularization(ModelParameters, numberOfData)
	
	local SquaredModelParameters 
	
	local AbsoluteModelParameters
	
	local SumSquaredModelParameters
	
	local SumAbsoluteModelParameters
	
	local regularizationValue
	
	if (self.regularizationMode == "L1") or (self.regularizationMode == "Lasso") then
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		if (self.hasBias) then AbsoluteModelParameters = makeLambdaAtBiasZero(AbsoluteModelParameters) end
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		regularizationValue = self.lambda * SumAbsoluteModelParameters
		
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		regularizationValue = self.lambda * SumSquaredModelParameters
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		local regularizationValuePart1 = self.lambda * SumSquaredModelParameters
		
		local regularizationValuePart2 = self.lambda * SumAbsoluteModelParameters
		
		regularizationValue = regularizationValuePart1 + regularizationValuePart2
		
	else
		
		error("Regularization Mode Does Not Exist!")
		
	end
	
	regularizationValue = regularizationValue / numberOfData
	
	return regularizationValue
	
end

return Regularization
