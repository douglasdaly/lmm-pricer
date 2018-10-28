/***********************************************************************************\
|																					|
|	LMM.cuh																			|
|																					|
|		Contains all functions required for generating LMM Monte Carlo Simulation	|
|	Paths for use in pricing IR Derivatives.										|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 5, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Pre-Processor Definitions
#ifndef _LMM_H_
#define _LMM_H_

//System Includes
#include <vector>
#include <iostream>
#include <stdlib.h>

//CUDA Includes
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//User Includes


//Definitions


//Function Declarations
// - CUDA LMM Kernel Functions
__global__ void setupRNGKernel(curandStatus_t* status, unsigned long seed);
__device__ double generateRNG(curandState* GlobalState, int Ind);
__global__ void generateLMMPath(double* Rates, double* Lambdas, double* Deltas, unsigned int Length, double* retPath);

// - Inline Functions for 2D and 3D Array Indexing
__host__ __device__ inline unsigned int index3d(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int NPaths, const unsigned int Length);
__host__ __device__ inline unsigned int index2d(const unsigned int x, const unsigned int y, const unsigned int Length);

//Namespaces
namespace LMM
{
	//High-Level Wrapper Functions for LMM MC Paths
	bool generateLMMSamplePaths(std::vector<double> Times, std::vector<double> Rates, std::vector<double> CapVols, unsigned int NPaths, double*** RetData);
	bool generateCPULMMSamplePaths(std::vector<double> Times, std::vector<double> Rates, std::vector<double> CapVols, unsigned int NPaths, double*** RetData);

	//Path Computation Functions called by High-Level Functions
	cudaError_t cudaGenerateLMMSamplePaths(double* rates, double* lambdas, double* deltas, unsigned int Length, unsigned int NPaths, double*** retData);
	void cpuGeneratePaths(std::vector<double*> trickVector1, std::vector<unsigned int> trickVector2);

	//Support Functions
	double calculateCapletPrice(double L, double K, double F, double T, double Sigma, double Delta, double PV);
	double calculateCapPrice(double L, double aK, double* aF, double* aT, double Sigma, double Delta, double* aPV, unsigned int Length);
	double ImplyCapletSpotVolatility(double& Time, double& Rate, double& Strike, double& Delta, double& PV, double& MarketPrice, double& Guess);
}

#endif