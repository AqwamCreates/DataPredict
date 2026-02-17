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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseSolver = require("Solver_BaseSolver")

local LevenbergMarquardtSolver = {}

LevenbergMarquardtSolver.__index = LevenbergMarquardtSolver

setmetatable(LevenbergMarquardtSolver, BaseSolver)

local defaultLambda = 1

function LevenbergMarquardtSolver.new(parameterDictionary)
	
	local NewLevenbergMarquardtSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewLevenbergMarquardtSolver, LevenbergMarquardtSolver)
	
	NewLevenbergMarquardtSolver:setName("LevenbergMarquardt")
	
	NewLevenbergMarquardtSolver.lambda = parameterDictionary.lambda or defaultLambda
	
	NewLevenbergMarquardtSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewLevenbergMarquardtSolver.isNonLinearInput)
		
		local pMatrix = (isLinearInput and NewLevenbergMarquardtSolver.cache)

		if (not pMatrix) then

			local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)

			pMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeMatrix)
			
			local pMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pMatrix)
			
			local diagonalMatrix = AqwamTensorLibrary:createIdentityTensor(pMatrixDimensionSizeArray, NewLevenbergMarquardtSolver.lambda)
			
			pMatrix = AqwamTensorLibrary:add(pMatrix, diagonalMatrix)

			pMatrix = AqwamTensorLibrary:inverse(pMatrix)
			
			-- If it is non-invertible, then do not return any weight change values as it is likely to be a local minimum.
			
			if (not pMatrix) then return AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(weightMatrix), 0) end

			pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFirstDerivativeMatrix)
			
			if (isLinearInput) then NewLevenbergMarquardtSolver.cache = pMatrix end

		end

		return AqwamTensorLibrary:dotProduct(pMatrix, firstDerivativeLossMatrix)
		
	end)
	
	return NewLevenbergMarquardtSolver
	
end

return LevenbergMarquardtSolver
