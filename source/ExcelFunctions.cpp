/***********************************************************************************\
|																					|
|	ExcelFunctions.cpp																|
|																					|
|		Contains all functions required for interfacing with MS Excel to C++.		|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 5, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Header Include
#include "ExcelFunctions.h"

//System Includes
#include <assert.h>
#include <iostream>
#include <vector>
#include <thread>

//CUDA Includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//User Includes
#include "LMM.cuh"

//Function Definitions

// - DLL Specific Functions

/*
 *	DLL Entry-point/Load Function
 */
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
	return TRUE;
}

// - LMM Functions
#pragma region LMM Monte Carlo Functions

/*
 *	Call from Excel to Generate LMM Simulation Paths on CPU
 */
int __stdcall GenerateCPULMMPaths(double* arrTimes, double* arrRates, double* arrVols, double* retData, int& ArrLength, int& NPaths)
{
	//Check if NPaths is too big
	int maxSims = 0;

	int NP = NPaths;

	int ret = 0;

	unsigned int nCores = std::thread::hardware_concurrency();

	maxSims = (1000 * nCores);

	if(maxSims < NP)
	{
		return maxSims;
	}

	std::vector<double> vTimes = std::vector<double>();
	std::vector<double> vRates = std::vector<double>();
	std::vector<double> vVols = std::vector<double>();

	for(int i=0; i<ArrLength; i++)
	{
		vTimes.push_back(arrTimes[i]);
		vRates.push_back(arrRates[i]);
		vVols.push_back(arrVols[i]);
	}

	//Setup Return Array
	double*** retArr = new double**[NP];
	for(int i=0; i<NP; i++)
	{
		retArr[i] = new double*[ArrLength];
		for(int j=0; j<ArrLength; j++)
		{
			retArr[i][j] = new double[ArrLength];
			for(int k=0; k<ArrLength; k++)
			{
				retArr[i][j][k] = 0.0;
			}
		}
	}
	
	//Do CUDA Calc

	bool result = LMM::generateCPULMMSamplePaths(vTimes, vRates, vVols, NP, retArr);

	if(!result)
	{
		ret = 1;
	}

	//Process results to return
	long h = 0;
	
	for(int i=0; i<NP; i++)
	{
		for(int j=0; j<ArrLength; j++)
		{
			retData[h] = i;
			++h;
			retData[h] = j;
			++h;

			for(int k=0; k<ArrLength; k++)
			{
				retData[h] = retArr[i][j][k];
				++h;
			}
		}
	}


	//Delete Results
	for(int i=0; i<NP; i++)
	{
		for(int j=0; j<ArrLength; j++)
		{
			delete [] retArr[i][j];
		}
		delete [] retArr[i];
	}
	delete [] retArr;

	//Return Results
	return ret;
}

/*
 *	Call from Excel to Generate LMM Simulation Paths on GPU
 */
int __stdcall GenerateCUDALMMPaths(double* arrTimes, double* arrRates, double* arrVols, double* retData, int& ArrLength, int& NPaths)
{
	//Check if NPaths is too big
	int maxSims = 0;

	int NP = NPaths;

	int ret = 0;

	cudaDeviceProp cdp;
	cudaGetDeviceProperties(&cdp, 0);

	maxSims = (cdp.maxThreadsPerMultiProcessor * cdp.multiProcessorCount);

	if(maxSims < NP)
	{
		return maxSims;
	}
	else if(maxSims == 0)
	{
		return -1;
	}

	std::vector<double> vTimes = std::vector<double>();
	std::vector<double> vRates = std::vector<double>();
	std::vector<double> vVols = std::vector<double>();

	for(int i=0; i<ArrLength; i++)
	{
		vTimes.push_back(arrTimes[i]);
		vRates.push_back(arrRates[i]);
		vVols.push_back(arrVols[i]);
	}

	//Setup Return Array
	double*** retArr = new double**[NP];
	for(int i=0; i<NP; i++)
	{
		retArr[i] = new double*[ArrLength];
		for(int j=0; j<ArrLength; j++)
		{
			retArr[i][j] = new double[ArrLength];
			for(int k=0; k<ArrLength; k++)
			{
				retArr[i][j][k] = 0.0;
			}
		}
	}
	
	//Do CUDA Calc

	bool result = LMM::generateLMMSamplePaths(vTimes, vRates, vVols, NP, retArr);

	if(!result)
	{
		ret = 1;
	}

	//Process results to return
	long h = 0;
	
	for(int i=0; i<NP; i++)
	{
		for(int j=0; j<ArrLength; j++)
		{
			retData[h] = i;
			++h;
			retData[h] = j;
			++h;

			for(int k=0; k<ArrLength; k++)
			{
				retData[h] = retArr[i][j][k];
				++h;
			}
		}
	}


	//Delete Results
	for(int i=0; i<NP; i++)
	{
		for(int j=0; j<ArrLength; j++)
		{
			delete [] retArr[i][j];
		}
		delete [] retArr[i];
	}
	delete [] retArr;

	//Return Results
	return ret;
}

#pragma endregion

#pragma region LMM Support Functions (Caps and Caplets)

 /*
  *	Calculate a Cap Price
  */
 double __stdcall CalculateCapPrice(double& L, double& Delta, double* arrTimes, double* arrRates, double& arrStrikes, double& flatVol, double* arrPVs, int& ArrLen)
 {
	 return LMM::calculateCapPrice(L, arrStrikes, arrRates, arrTimes, flatVol, Delta, arrPVs, ArrLen);
 }

 /*
  *	Imply Caplet Volatility using Secant Method (Precision 0.00001 & Max Iterations = 1000)
  */
 double __stdcall ImplyCapletVolatility(double& Time, double& Rate, double& Strike, double& Delta, double& PV, double& MarketPrice, double& Guess)
 {
	 return LMM::ImplyCapletSpotVolatility(Time, Rate, Strike, Delta, PV, MarketPrice, Guess);
 }

#pragma endregion