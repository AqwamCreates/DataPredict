--[[

	--------------------------------------------------------------------

	Version 1.3.1

	Aqwam's Roblox Deep Learning Library (AR-MDLL)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
	
	By using this library, you agree to comply with our Terms and Conditions in the link below:

	https://github.com/AqwamCreates/Aqwam-Roblox-Machine-And-Deep-Learning-Library/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local requiredMatrixLibraryVersion = 1.5

local AqwamMatrixLibrary = require(script.AqwamRobloxMatrixLibraryLinker.Value)

local Models = script.Models

local LinearRegression = require(Models.LinearRegression)
local LogisticRegression = require(Models.LogisticRegression)
local KMeans = require(Models.KMeans)
local SupportVectorMachine = require(Models.SupportVectorMachine)
local ExpectationMaximization = require(Models.ExpectationMaximization)
local NaiveBayes = require(Models.NaiveBayes)
local NeuralNetwork = require(Models.NeuralNetwork)

local Others = script.Others

local ModelChecking = require(Others.ModelChecking) -- for testing and validating datasets
local GradientDescendModes = require(Others.GradientDescentModes)
local Regularization = require(Others.Regularization)

local Optimizers = script.Optimizers

local RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation)
local Momentum = require(Optimizers.Momentum)
local AdaptiveGradient = require(Optimizers.AdaptiveGradient)
local AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation)

local ModelCheckingDictionary = {

	testRegressionModel = ModelChecking.testRegressionModel,
	testLogisticModel = ModelChecking.testClassificationModel

}

local ModelsDictionary = {
	
	LinearRegression = LinearRegression,
	LogisticRegression = LogisticRegression,
	KMeans = KMeans,
	SupportVectorMachine = SupportVectorMachine,
	NaiveBayes = NaiveBayes,
	ExpectationMaximization = ExpectationMaximization,
	NeuralNetwork = NeuralNetwork,
	
}

local OptimizersDictionary = {
	
	RootMeanSquarePropagation = RootMeanSquarePropagation,
	Momentum = Momentum,
	AdaptiveGradient = AdaptiveGradient,
	AdaptiveMomentEstimation = AdaptiveMomentEstimation
	
}

local OthersDictionary = {
	
	GradientDescendModes = GradientDescendModes,
	ModelChecking = ModelChecking,
	Regularization = Regularization
	
}

local AqwamRobloxMachineLearningLibrary = {}

AqwamRobloxMachineLearningLibrary.Models =  ModelsDictionary

AqwamRobloxMachineLearningLibrary.Optimizers = OptimizersDictionary

AqwamRobloxMachineLearningLibrary.Others = OthersDictionary

local function checkVersion()
	
	local matrixLibraryVersion
	
	local success = pcall(function()
		
		matrixLibraryVersion = AqwamMatrixLibrary:getVersion()
		
	end)
	
	if not success then matrixLibraryVersion = -1 end
	
	if (matrixLibraryVersion < requiredMatrixLibraryVersion) then warn("The matrix library is out-of-date. You may encounter some problems.") end
	
end

checkVersion()

return AqwamRobloxMachineLearningLibrary

local RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation)
local Momentum = require(Optimizers.Momentum)
local AdaptiveGradient = require(Optimizers.AdaptiveGradient)
local AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation)

local ModelCheckingDictionary = {

	testRegressionModel = ModelChecking.testRegressionModel,
	testLogisticModel = ModelChecking.testClassificationModel

}

local ModelsDictionary = {
	
	LinearRegression = LinearRegression,
	LogisticRegression = LogisticRegression,
	KMeans = KMeans,
	SupportVectorMachine = SupportVectorMachine,
	NaiveBayes = NaiveBayes,
	ExpectationMaximization,
	
}

local OptimizersDictionary = {
	
	RootMeanSquarePropagation = RootMeanSquarePropagation,
	Momentum = Momentum,
	AdaptiveGradient = AdaptiveGradient,
	AdaptiveMomentEstimation = AdaptiveMomentEstimation
	
}

local OthersDictionary = {
	
	GradientDescendModes = GradientDescendModes,
	ModelChecking = ModelChecking,
	Regularization = Regularization
	
}

local AqwamRobloxMachineLearningLibrary = {}

AqwamRobloxMachineLearningLibrary.Models =  ModelsDictionary

AqwamRobloxMachineLearningLibrary.Optimizers = OptimizersDictionary

AqwamRobloxMachineLearningLibrary.Others = OthersDictionary

local function checkVersion()
	
	local matrixLibraryVersion
	
	local success = pcall(function()
		
		matrixLibraryVersion = AqwamMatrixLibrary:getVersion()
		
	end)
	
	if not success then matrixLibraryVersion = -1 end
	
	if (matrixLibraryVersion < requiredMatrixLibraryVersion) then warn("The matrix library is out-of-date. You may encounter some problems.") end
	
end

checkVersion()

return AqwamRobloxMachineLearningLibrary
