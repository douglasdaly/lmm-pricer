/***********************************************************************************\
|																					|
|	MathHelper.h																	|
|																					|
|		Contains all functions for math functions used for finance, such as CDF		|
|	and PDF for Normal Distribution, Integration, Differentiation, Optimization		|
|	and equation solvers.  CPU Version												|
|																					|
|	Author:		Douglas James Daly Jr.												|
|	Date:		April 5, 2014														|
|	Version:	0.2.1																|
|																					|
\***********************************************************************************/

//Pre-Processor
#ifndef _MATHHELPER_H_
#define _MATHHELPER_H_

//System Includes


//Definitions


//Function Prototypes


//Namespaces
namespace MathHelper
{
	//Probability Functions
	double StdNormalCDF(double x);
	double StdNormalPDF(double x);

}


#endif