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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseSolver = require(script.Parent.BaseSolver)

local ConjugateGradientSolver = {}

ConjugateGradientSolver.__index = ConjugateGradientSolver

setmetatable(ConjugateGradientSolver, BaseSolver)

function ConjugateGradientSolver.new(parameterDictionary)
	
	local NewConjugateGradientSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewConjugateGradientSolver, ConjugateGradientSolver)
	
	NewConjugateGradientSolver:setName("ConjugateGradient")
	
	NewConjugateGradientSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewConjugateGradientSolver.isNonLinearInput)
		
		local aMatrix = (isLinearInput and NewConjugateGradientSolver.cache)
		
		local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)
		
		if (not aMatrix) then

			aMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeMatrix)

			if (isLinearInput) then NewConjugateGradientSolver.cache = aMatrix end

		end
		
		local unaryFirstDerivativeLossMatrix = AqwamTensorLibrary:unaryMinus(firstDerivativeLossMatrix)
		
		local weightChangeMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, unaryFirstDerivativeLossMatrix)
		
		-- Using weightMatrix here as the initial guess for online learning.
		
		local residualMatrix = AqwamTensorLibrary:subtract(weightChangeMatrix, AqwamTensorLibrary:dotProduct(aMatrix, weightMatrix))
		
		local weightDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix)

		local maximumNumberOfIterations = weightDimensionSizeArray[1] * weightDimensionSizeArray[2]
		
		local originalWeightMatrix = weightMatrix
		
		local pMatrix = residualMatrix
		
		local numberOfIterations = 0
		
		local transposedResidualMatrix
		
		local alphaNumeratorMatrix
		
		local alphaDenominatorMatrix
		
		local alphaMatrix
		
		local transposedPMatrix
		
		local residualChangeMatrix
		
		local newResidualMatrix
		
		local transposedNewResidualMatrix
		
		local betaNumeratorMatrix
		
		local betaDenominatorMatrix
		
		local betaMatrix
		
		local pChangeMatrix
		
		local previousResidualScoreValue = math.huge
		
		local residualScoreValue
		
		repeat
			
			numberOfIterations = numberOfIterations + 1
			
			transposedResidualMatrix = AqwamTensorLibrary:transpose(residualMatrix)
			
			transposedPMatrix = AqwamTensorLibrary:transpose(pMatrix)
			
			alphaNumeratorMatrix = AqwamTensorLibrary:dotProduct(transposedResidualMatrix, residualMatrix)
			
			alphaDenominatorMatrix = AqwamTensorLibrary:dotProduct(transposedPMatrix, aMatrix, pMatrix)
			
			alphaMatrix =  AqwamTensorLibrary:divide(alphaNumeratorMatrix, alphaDenominatorMatrix)
			
			weightChangeMatrix = AqwamTensorLibrary:multiply(alphaMatrix, pMatrix)
			
			weightMatrix = AqwamTensorLibrary:add(weightMatrix, weightChangeMatrix)
			
			residualChangeMatrix = AqwamTensorLibrary:multiply(alphaMatrix, pMatrix)
			
			newResidualMatrix = AqwamTensorLibrary:subtract(residualMatrix, residualChangeMatrix)
			
			transposedNewResidualMatrix = AqwamTensorLibrary:transpose(newResidualMatrix)
			
			betaNumeratorMatrix = AqwamTensorLibrary:dotProduct(transposedNewResidualMatrix, newResidualMatrix)
			
			betaDenominatorMatrix = AqwamTensorLibrary:dotProduct(transposedResidualMatrix, residualMatrix)[1][1]
			
			betaMatrix = AqwamTensorLibrary:divide(betaNumeratorMatrix, betaDenominatorMatrix)
			
			pChangeMatrix = AqwamTensorLibrary:dotProduct(aMatrix, pMatrix)
			
			pChangeMatrix = AqwamTensorLibrary:multiply(pChangeMatrix, betaMatrix)
			
			pMatrix = AqwamTensorLibrary:add(newResidualMatrix, pChangeMatrix)
			
			residualScoreValue = AqwamTensorLibrary:sum(residualMatrix)
			
			if (residualScoreValue == previousResidualScoreValue) then break end
			
			previousResidualScoreValue = residualScoreValue
			
		until (numberOfIterations == maximumNumberOfIterations) 
		
		return AqwamTensorLibrary:subtract(weightMatrix, originalWeightMatrix) -- To apply optimizer later.
		
	end)
	
	return NewConjugateGradientSolver
	
end

return ConjugateGradientSolver
