/***********************************************************************************\
|																					|
|	LMM.cu																			|
|																					|
|		Contains all functions required for generating LMM Monte Carlo Simulation	|
|	Paths for use in pricing IR Derivatives.										|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 5, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Header Include
#include "LMM.cuh"

//System Includes
#include <ctime>
#include <mutex>
#include <thread>
#include <random>
#include <math.h>

#include <assert.h>

//User Includes
#include "MathHelper.h"

//Definitions

//Function Definitions

#pragma region CUDA LMM Kernel Functions
/*
 *	Initialize the CURAND for the Kernels
 */
__global__ void setupRNGKernel(curandState* state, unsigned long seed, unsigned int NPaths)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= NPaths)
	{
		return;
	}

	curand_init(seed*((unsigned long)(idx+1)*(idx+1)), idx, 0, &state[idx]);
}

/*
 *	Use CURAND to generate a random normal
 */
__device__ double generateRNG(curandState* GlobalState, int ind)
{
	curandState localState = GlobalState[ind];
	double RANDOM = curand_normal_double(&localState);
	GlobalState[ind] = localState;
	return RANDOM;
}

/*
 *	CUDA Fn. to generate LMM Path
 */
__global__ void generateLMMPath(double* Rates, double* Lambdas, double* Deltas, unsigned int Length, unsigned int NPaths, curandState* GlobalState, double* retPath)
{
	//Get Thread Index
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= NPaths)
	{
		return;
	}

	// - Starting  Rates
	for(unsigned int j = 0; j < Length; j++)
	{
		retPath[index3d(idx, 0, j, NPaths, Length)] = Rates[j];
	}

	//Now we need to do the rest
	for(unsigned int j=1; j<Length; j++)
	{
		//Cycle through t_j's
		/*
		// - Zero out unused
		for(int k=0; k<j; k++)
		{
			retPath[index3d(idx, j, k, NPaths, Length)] = 0.0;
		}
		*/

		// - Now calculate the relevant forward rates
		for(int k=j; k<Length; k++)
		{
			//Get F_kj
			double Fkj = retPath[index3d(idx, j-1, k, NPaths, Length)];
			
			double expSum = 0.0;
			double expF2 = 0.0;

			double Ljk = Lambdas[index2d(j, k, Length)];
			double Dj1 = Deltas[j-1];

			//Inner sum
			for(int i=j; i<k; i++)
			{
				double Fij = retPath[index3d(idx, j-1, i, NPaths, Length)];

				expSum += ((Deltas[i]*Fij*Lambdas[index2d(j,i,Length)]*Ljk)/(1+Deltas[i]*Fij));
			}

			expSum -= (pow(Ljk,2.0)/2.0);

			expSum *= Dj1;

			expF2 = Ljk*sqrt(Dj1);
			
			double rn = generateRNG(GlobalState, idx);

			expF2 *= rn;

			//Save
			retPath[index3d(idx, j,  k, NPaths, Length)] = Fkj*exp(expSum + expF2);
		}
	}
}

#pragma endregion

//Namespace Definitions

#pragma region LMM Monte Carlo Simulation Functions [CPU]
/*
 *	Generate LMM Sample Paths Function (CPU)
 */
bool LMM::generateCPULMMSamplePaths(std::vector<double> Times, std::vector<double> Rates, std::vector<double> CapVols, unsigned int NPaths, double*** RetData)
{
	//First we need to generate/calculate the Lambda values
	unsigned int len = Times.size();

	std::vector<double> Lambdas;
	std::vector<double> Deltas;

	//Calculate the Deltas
	for(int i=0; i<len; i++)
	{
		if(i == 0)
		{
			Deltas.push_back(Times[0]);
		}
		else{
			Deltas.push_back(Times[i] - Times[i-1]);
		}
	}


	/*
	//Calculate the Lambdas (assuming time-homogeneity)
	for(int i=0; i<len; i++)
	{
		double iSigma = CapVols[i];
		double iTime = Times[i];
		double iDelta = 0.25;
		double iLambda = 0.0;

		double lhs = (pow(iSigma, 2.0))*iTime;
		double rhs = 0.0;

		if(i == 0)
		{
			Lambdas.push_back(iSigma);
			continue;
		}

		for(int j=0; j<i; j++)
		{
			rhs += (pow(Lambdas[j], 2.0)*(Deltas[j]));
		}

		iLambda = sqrt((lhs-rhs)/iDelta);
			
		//Save Result
		Lambdas.push_back(iLambda);
	}
	*/

	// Calculate the Lambdas - Crude (for non-time homogeneity)
	for(int i=0; i<len; i++)
	{
		Lambdas.push_back(CapVols[i]);
	}

	//Generate Lambda Matrix
	double* mLambdas = new double[len*len];

	for(int i=0; i<len; i++)
	{
		for(int j=0; j<i; j++)
		{
			mLambdas[index2d(i, j, len)] = 0.0;
		}
		for(int j=i; j<len; j++)
		{
			mLambdas[index2d(i, j, len)] = Lambdas[j-i];
		}
	}

	//Now we need to format the data for CUDA
	double* aRates = (double*) malloc(len * sizeof(double));
	double* aDeltas = (double*) malloc(len * sizeof(double));

	for(int i=0; i<len; i++)
	{
		aRates[i] = Rates[i];
		aDeltas[i] = Deltas[i];
	}

	//CALCULATIONS FOR LMM
	// - Format the return data for the function
	double* d_rdata;

	d_rdata = new double[len*len*NPaths];
	
	// - Zero out bytes
	for(int i=0; i<(len*len*NPaths); i++)
	{
		d_rdata[i] = 0;
	}


	// - Get Max Threads
	unsigned int maxThreads = std::thread::hardware_concurrency();

	if(NPaths < maxThreads)
	{
		maxThreads = 1;
	}

	// - Divy up the indices for the threads
	std::vector<unsigned int> threadIndices;

	int baseIdxPerThread = ((NPaths - (NPaths % maxThreads))/maxThreads);
	int remainder = NPaths % maxThreads;

	for(int i=0; i<maxThreads; i++)
	{
		threadIndices.push_back((i+1)*baseIdxPerThread);

		if(i == maxThreads - 1)
		{
			threadIndices[i] += remainder;
			threadIndices[i] -= 1;
		}
	}

	// - Launch the Threads
	std::thread* tPtr = new std::thread[maxThreads];

	for(int i=0; i<maxThreads; i++)
	{
		//Launch thread
		unsigned int sIdx, eIdx;

		if(i == 0)
		{
			sIdx = 0;
			eIdx = threadIndices[0];
		}
		else{
			sIdx = threadIndices[i-1]+1;
			eIdx = threadIndices[i];
		}

		//Trick vectors due to arg limitations
		std::vector<double*> trickVec1;
		trickVec1.push_back(aRates);
		trickVec1.push_back(mLambdas);
		trickVec1.push_back(aDeltas);
		trickVec1.push_back(d_rdata);

		std::vector<unsigned int> trickVec2;
		trickVec2.push_back(len);
		trickVec2.push_back(NPaths);
		trickVec2.push_back(sIdx);
		trickVec2.push_back(eIdx);

		tPtr[i] = std::thread(&LMM::cpuGeneratePaths, trickVec1, trickVec2);

	}

	//Join em when they're all done
	for(int i=0; i<maxThreads; i++)
	{
		tPtr[i].join();
	}

	//END LMM CALCULATIONS

	//Clean up
	free(aRates);
	free(aDeltas);
	delete [] mLambdas;

	delete [] tPtr;

	//Format ret data
	//Return the Data to the Host
	for(int i=0; i<NPaths; i++)
	{
		for(int j=0; j<len; j++)
		{
			for(int k=0; k<len; k++)
			{
				RetData[i][j][k] = d_rdata[index3d(i, j, k, NPaths, len)];
			}
		}
	}

	delete [] d_rdata;

	//return
	return true;
}

/*
 *	Generate a Set of sample paths on the CPU for the given indices
 */
void LMM::cpuGeneratePaths(std::vector<double*> trickVector1, std::vector<unsigned int> trickVector2)
{
	//Extract data from trick vectors
	double* Rates = trickVector1[0];
	double* Lambdas = trickVector1[1];
	double* Deltas = trickVector1[2];
	double* retPath = trickVector1[3];

	unsigned int Length = trickVector2[0];
	unsigned int NPaths = trickVector2[1];
	unsigned int startIdx = trickVector2[2];
	unsigned int endIdx = trickVector2[3];

	//Setup RNG Engine
	std::default_random_engine generator(endIdx*time(NULL));
	std::normal_distribution<double> ndist(0.0, 1.0);

	//Run Loops/Calcs
	for(int idx=startIdx; idx<=endIdx; idx++)
	{
		// - Starting  Rates
		for(unsigned int j = 0; j < Length; j++)
		{
			retPath[index3d(idx, 0, j, NPaths, Length)] = Rates[j];
		}

		//Now we need to do the rest
		for(unsigned int j=1; j<Length; j++)
		{
			//Cycle through t_j's
			// - Zero out unused
			for(int k=0; k<j; k++)
			{
				retPath[index3d(idx, j, k, NPaths, Length)] = 0.0;
			}

			// - Now calculate the relevant forward rates
			for(int k=j; k<Length; k++)
			{
				//Get F_kj
				double Fkj = retPath[index3d(idx, j-1, k, NPaths, Length)];

				double expSum = 0.0;
				double expF2 = 0.0;

				//Inner sum
				for(int i=j; i<k; i++)
				{
					double Fij = retPath[index3d(idx, j-1, i, NPaths, Length)];

					expSum += ((Deltas[i]*Fij*Lambdas[index2d(j,i,Length)]*Lambdas[index2d(j,k,Length)])/(1+Deltas[i]*Fij));
				}

				expSum -= (pow(Lambdas[index2d(j, k, Length)],2.0)/2.0);

				expSum *= Deltas[j-1];

				expF2 = Lambdas[index2d(j,k,Length)]*sqrt(Deltas[j-1]);
			
				double rn = ndist(generator);

				expF2 *= rn;

				//Save
				retPath[index3d(idx, j,  k, NPaths, Length)] = Fkj*exp(expSum + expF2);
			}
		}
	}
}

#pragma endregion

#pragma region LMM Monte Carlo Simulation Functions [GPU]

/*
 *	Generate LMM Sample Paths Function (GPU)
 */
bool LMM::generateLMMSamplePaths(std::vector<double> Times, std::vector<double> Rates, std::vector<double> CapVols, unsigned int NPaths, double*** RetData)
{
	//First we need to generate/calculate the Lambda values
	unsigned int len = Times.size();

	std::vector<double> Lambdas;
	std::vector<double> Deltas;

	//Calculate the Deltas
	for(int i=0; i<len; i++)
	{
		if(i == 0)
		{
			Deltas.push_back(Times[0]);
		}
		else{
			Deltas.push_back(Times[i] - Times[i-1]);
		}
	}


	/*
	//Calculate the Lambdas (assuming time-homogeneity)
	for(int i=0; i<len; i++)
	{
		double iSigma = CapVols[i];
		double iTime = Times[i];
		double iDelta = 0.25;
		double iLambda = 0.0;

		double lhs = (pow(iSigma, 2.0))*iTime;
		double rhs = 0.0;

		if(i == 0)
		{
			Lambdas.push_back(iSigma);
			continue;
		}

		for(int j=0; j<i; j++)
		{
			rhs += (pow(Lambdas[j], 2.0)*(Deltas[j]));
		}

		iLambda = sqrt((lhs-rhs)/iDelta);
			
		//Save Result
		Lambdas.push_back(iLambda);
	}
	*/

	// Calculate the Lambdas - Crude (for non-time homogeneity)
	for(int i=0; i<len; i++)
	{
		Lambdas.push_back(CapVols[i]);
	}

	//Generate Lambda Matrix
	double* mLambdas = new double[len*len];

	for(int i=0; i<len; i++)
	{
		for(int j=0; j<i; j++)
		{
			mLambdas[index2d(i, j, len)] = 0.0;
		}
		for(int j=i; j<len; j++)
		{
			mLambdas[index2d(i, j, len)] = Lambdas[j-i];
		}
	}

	//Now we need to format the data for CUDA
	double* aRates = (double*) malloc(len * sizeof(double));
	double* aDeltas = (double*) malloc(len * sizeof(double));

	for(int i=0; i<len; i++)
	{
		aRates[i] = Rates[i];
		aDeltas[i] = Deltas[i];
	}

	//Generate
	cudaError_t cudaErr = cudaGenerateLMMSamplePaths(aRates, mLambdas, aDeltas, len, NPaths, RetData);

	cudaError_t cudaErr2 = cudaDeviceReset();

	//Clean-up
	free(aRates);
	free(aDeltas);
	delete [] mLambdas;

	if(cudaErr != cudaSuccess)
	{
		//Error Occurred
		std::cout << "CUDA Error: " << cudaGetErrorString(cudaErr) << std::endl;
		return false;
	}
	else{
		//Success
		return true;
	}
}

/*
 *	CUDA Function to generate the paths
 */
cudaError_t LMM::cudaGenerateLMMSamplePaths(double* rates, double* lambdas, double* deltas, unsigned int Length, unsigned int NPaths, double*** retData)
{
	//Move Data to Device Memory
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);

	if(cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	double* d_rates;
	double* d_lambdas;
	double* d_deltas;

	//Allocate Memory
	cudaStatus = cudaMalloc((void**)&d_rates, Length * sizeof(double));
	
	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMalloc(d_rates)." << std::endl;
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_lambdas, Length * Length * sizeof(double));

	if(cudaStatus != cudaSuccess)
	{
		cudaFree(d_rates);
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMalloc(d_lambdas)." << std::endl;
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_deltas, Length * sizeof(double));

	if(cudaStatus != cudaSuccess)
	{
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMalloc(d_deltas)." << std::endl;
		return cudaStatus;
	}

	//Move Data from Host to Device
	bool success = true;

	cudaStatus = cudaMemcpy(d_rates, rates, Length * sizeof(double), cudaMemcpyHostToDevice);

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMemcpy(d_rates)." << std::endl;
		success &= false;
	}
	
	cudaStatus = cudaMemcpy(d_lambdas, lambdas, Length * Length * sizeof(double), cudaMemcpyHostToDevice);

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMemcpy(d_lambdas)." << std::endl;
		success &= false;
	}

	cudaStatus = cudaMemcpy(d_deltas, deltas, Length * sizeof(double), cudaMemcpyHostToDevice);

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMemcpy(d_deltas)." << std::endl;
		success &= false;
	}

	if(!success)
	{
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		return cudaStatus;
	}

	//Return-Data Setup
	double* d_rdata;

	cudaStatus = cudaMalloc((void**)&d_rdata, NPaths * Length * Length * sizeof(double));

	if(cudaStatus != cudaSuccess)
	{
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMalloc(d_rdata)." << std::endl;
		return cudaStatus;
	}

	// - Zero out bytes
	for(int i=0; i<NPaths*Length*Length; i++)
	{
		cudaStatus = cudaMemset(&d_rdata[i], 0, sizeof(double));
	}
		
	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMemset(d_rdata)." << std::endl;
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		cudaFree(d_rdata);
		return cudaStatus;
	}

	//Setup RNG Stuff
	curandState* devStates;
	cudaStatus = cudaMalloc(&devStates, NPaths * sizeof(curandState));

	if(cudaStatus != cudaSuccess)
	{
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		cudaFree(d_rdata);
		std::cout << "CUDA Error: " << cudaStatus << " on cudaMalloc(devStates)." << std::endl;
		return cudaStatus;
	}

	//Run it
	unsigned int M_THREADS, M_BLOCKS;

	M_THREADS = 256;

	M_BLOCKS = (unsigned int)(((NPaths+M_THREADS-1)/M_THREADS));

	unsigned long seedTime = (unsigned) time(NULL);

	setupRNGKernel <<<M_BLOCKS,M_THREADS>>> (devStates, seedTime, NPaths);

	cudaStatus = cudaDeviceSynchronize();

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaDeviceSynchronize(). [1]" << std::endl;
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		cudaFree(d_rdata);
		cudaFree(devStates);
		return cudaStatus;
	}

	generateLMMPath <<<M_BLOCKS,M_THREADS>>> (d_rates, d_lambdas, d_deltas, Length, NPaths, devStates, d_rdata);

	cudaStatus = cudaDeviceSynchronize();

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "CUDA Error: " << cudaStatus << " on cudaDeviceSynchronize(). [2]" << std::endl;
		cudaFree(d_rates);
		cudaFree(d_lambdas);
		cudaFree(d_deltas);
		cudaFree(d_rdata);
		cudaFree(devStates);
		return cudaStatus;
	}

	//Return the Data to the Host
	for(int i=0; i<NPaths; i++)
	{
		for(int j=0; j<Length; j++)
		{
			for(int k=0; k<Length; k++)
			{
				cudaMemcpy(&retData[i][j][k], &d_rdata[index3d(i, j, k, NPaths, Length)], sizeof(double), cudaMemcpyDeviceToHost);
			}
		}
	}

	cudaFree(d_rdata);
	cudaFree(d_rates);
	cudaFree(d_lambdas);
	cudaFree(d_deltas);

	cudaFree(devStates);

	return cudaStatus;
}

#pragma endregion

#pragma region Indexing Inline Functions

/*
 *	Inline function to convert 3D (x,y,z) index to 1D array index
 */
__host__ __device__ inline unsigned int index3d(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int NPaths, const unsigned int Length)
{
	return ((x * Length * Length) + (y * Length) + z);
}

/*
 *	Inline function to convert 2D (x,y) index to 1D array index
 */
__host__ __device__ inline unsigned int index2d(const unsigned int x, const unsigned int y, const unsigned int Length)
{
	return ((x*Length) + y);
}

#pragma endregion

#pragma region Cap and Caplet Functions

/*
 *	Calculate the price of a caplet
 */
double LMM::calculateCapletPrice(double L, double K, double F, double T, double Sigma, double Delta, double PV)
{
	double d1 = ((log(F/K)+(pow(Sigma, 2.0)*(T/2.0)))/(Sigma*sqrt(T)));
	double d2 = d1 - (Sigma*sqrt(T));

	return (L*Delta*PV*((F*MathHelper::StdNormalCDF(d1))-(K*MathHelper::StdNormalCDF(d2))));
}

/*
 *	Calculate the Price of a Cap
 */
double LMM::calculateCapPrice(double L, double aK, double* aF, double* aT, double Sigma, double Delta, double* aPV, unsigned int Length)
{
	double capPrice = 0.0;

	for(unsigned int i=0; i<Length; i++)
	{
		capPrice += LMM::calculateCapletPrice(L, aK, aF[i], aT[i], Sigma, Delta, aPV[i]);
	}

	return capPrice;
}

/*
 *	Imply Spot Caplet Volatility
 */
double LMM::ImplyCapletSpotVolatility(double& Time, double& Rate, double& Strike, double& Delta, double& PV, double& MarketPrice, double& Guess)
{
	//Using the Secant Method
	 unsigned int MAX_ITERS = 1000;
	 double PRECISION = 0.0001;

	 unsigned int iter = 0;
	 double x0 = Guess;
	 double x1 = Guess;
	 double x2 = 0.0;
	 double diff = 1.0;
	 double y0 = 0.0;
	 double y1 = 1.0;

	 do{
		if(iter == 0)
		{	 
			if(x0 == x1)
			{
				if(Time > 2.0)
				x1 = 0.9*x0;
				else
				x1 = 1.1*x0;
			}
		}
		
		x2 = x1 - (y1 * ( (x1 - x0) / (y1 - y0) ));
		x0 = x1;
		x1 = x2;
		

		y0 = y1;
		y1 = MarketPrice-LMM::calculateCapletPrice(1.0, Strike, Rate, Time, x1, Delta, PV);

		if(iter > 2)
		{
			diff = abs((y1 - y0)/y0);
		}

		++iter;
	 }while(diff >= PRECISION && iter <= MAX_ITERS);

	 return x1;
}

#pragma endregion