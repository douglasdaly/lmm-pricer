/***********************************************************************************\
|																					|
|	ExcelFunctions.h																|
|																					|
|		Contains all functions required for interfacing with MS Excel from C++.		|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 5, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Pre-Processor Declarations
#ifndef _EXCEL_FUNCTIONS_H_
#define _EXCEL_FUNCTIONS_H_

//Cross-Platform System Includes
#include <iostream>

//OS Specific Includes
#include <Windows.h>

//User Includes


//Definitions


//Function Prototypes

// - DLL Entry-point function
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved);

// - Functions called from Excel for Generating LMM Sample Paths on either GPU or CPU
int __stdcall GenerateCUDALMMPaths(double* arrTimes, double* arrRates, double* arrVols, double* retData, int& ArrLen, int& NPaths);
int __stdcall GenerateCPULMMPaths(double* arrTimes, double* arrRates, double* arrVols, double* retData, int& ArrLen, int& NPaths);

// - Functions called from Excel for 'support' - e.g. Caplet Implied Vol. Calc., Cap Pricing, etc.
double __stdcall CalculateCapPrice(double& L, double& Delta, double* arrTimes, double* arrRates, double& arrStrikes, double& flatVol, double* arrPVs, int& ArrLen);
double __stdcall ImplyCapletVolatility(double& Time, double& Rate, double& Strike, double& Delta, double& PV, double& MarketPrice, double& Guess);

#endif