/***********************************************************************************\
|																					|
|	MathHelper.cpp																	|
|																					|
|		Contains all functions for math functions used for finance, such as CDF		|
|	and PDF for Normal Distribution, Integration, Differentiation, Optimization		|
|	and equation solvers.  CPU Version												|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 8, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Pre-processor
#define _USE_MATH_DEFINES

//Header Include
#include "MathHelper.h"

//System Includes
#include <math.h>

//User Includes


//Function Definitions


//Namespace Function Definitions

#pragma region Probability Distribution Functions

/*
 *	PDF Function for Normal (Gaussian) Distribution
 */
double MathHelper::StdNormalPDF(double x)
{
	return (1.0/(sqrt(2.0*M_PI)))*exp(-((pow(x,2.0))/(2.0)));
}

/*
 *	CDF Function for Normal (Guassian) Distribution
 *		- Use approximation for speed
 *		- (Source: www.johndcook.com/cpp_phi.html)
 */
double MathHelper::StdNormalCDF(double x)
{
	//Constants
	double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

	//Save sign of x
	int sign = 1;
	if(x < 0)
	{
		sign = -1;
	}

	x = fabs(x)/sqrt(2.0);

	//A&S Formula 7.1.26
	double t = 1.0/(1.0 + p * x);
	double y = 1.0 - ((((((a5 * t + a4)*t) + a3)*t) + a2)*t + a1)*t*exp(-x*x);

	return (0.5*(1.0+sign*y));
}


#pragma endregion