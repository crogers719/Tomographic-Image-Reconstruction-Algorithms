/**********HEADERS**********/

#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>

#include "cuda_ptr.cuh"
#include "mimo-io.cuh"
using namespace std;

/**********DEFINING CONSTANTS***********/

#define NX 192				//was 201
#define NY 192				//was 201
#define NT 401

#define HX 0.001f
#define HY 0.001f
#define H 0.001f

#define DT 3.3333e-07f
#define OMEGAC 7.8540e+05f
#define TAO 4.0000e-06f
#define TT 8.1573e-06f

/**********FUNCTION DECLARATION**********/

//Host Functions
void Ultrasonic_Tomography(int);

//In-Line Functions
inline int grid_size(int, int);

//Device Functions
__global__ void field_setup(kernel_ptr<float> const, kernel_ptr<float> const, kernel_ptr<float>);


/***************MAIN PROGRAM***************/

int main(int argc, char **argv)
{
	// Time Measuring Variables
	int ti = 0, tf = 0;

	// set floating-point precision on stdout and stderr
	cout << fixed << setprecision(10);
	cerr << fixed << setprecision(10);

	cerr << "Ultrasonic Tomography Running:\n\n";

	//Initial time
	ti = clock();
	cerr << "ti = " << ti << "\n";

	Ultrasonic_Tomography(ti);
	cudaDeviceReset();

	//Calculate total time
	tf = clock();
	cerr << "tf = " << tf << "\n"
		 << "tt = " << tf - ti << "\n"
		 << "Total Seconds = " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";
}



/**********HOST FUNCTION DEFINITIONS**********/

void Ultrasonic_Tomography(int ti)
{
	// environment initialization

	host_ptr<float> x(NX);
	host_ptr<float> y(NY);
	device_ptr<float> dev_x(NX);
	device_ptr<float> dev_y(NY);

	for (int i = 0; i < NX; i++)
		x(i) = -0.1f + i * HX;

	for (int j = 0; j < NY; j++)
		y(j) = -0.1f + j * HY;

	copy(dev_x, x);
	copy(dev_y, y);

	// fo(i, j) =
	//    ground truth value at pos (i, j) of field
	host_ptr<float> fo(NX, NY);
	device_ptr<float> dev_fo(NX, NY);

	// kernel launch parameters for field kernels
	dim3 threads_field(NX, 1);
	dim3 grid_field(
		grid_size(NX, threads_field.x),
		grid_size(NY, threads_field.y));

	// initialize the ground truth field
	field_setup<<<grid_field, threads_field>>>(dev_x, dev_y, dev_fo);

	// copy from device to host
	copy(fo, dev_fo);

	cerr << "writing to 'fo.txt'...\n\n";

	ofstream fo_out("fo.txt");
	write(fo_out, fo);
}


/**********DEVICE FUNCTION DEFINITIONS***********/
__global__ void field_setup(kernel_ptr<float> const x, kernel_ptr<float> const y, kernel_ptr<float> fo)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		float value = 0.f;

		/* if(((sqrtf(powf(x(i) - 0.015f, 2.0f) + powf(y(j) + 0.000f, 2.0f))) <= 0.005f) || ((sqrtf(powf(x(i) + 0.015f, 2.0f) + powf(y(j) + 0.000f, 2.0f))) <= 0.005f)) */
		/* { */
		/* 	value = 0.06f; */
		/* } */
		/* else */
		/* { */
		/* 	if(sqrtf(x(i) * x(i) + y(j) * y(j)) <= 0.03f) */
		/* 	{ */
		/* 		value = 0.02f; */
		/* 	} */
		/* 	else */
		/* 	{ */
		/* 		value = 0.0f; */
		/* 	} */
		/* } */

		float rc = 0.015f;
		float rp = 0.005f;

		float sc = 0.03f;
		float sp = 0.05f;

		if (powf(x(i), 2) + powf(y(j), 2) <= powf(rc, 2))
		{
			value = sc;
		}

		if (powf(x(i) - rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y(j) - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2))
		{
			value = sp;
		}

		if (powf(x(i) + rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y(j) - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2))
		{
			value = sp;
		}

		if (powf(x(i), 2) + powf(y(j) + rc, 2) <= powf(rp, 2))
		{
			value = sp;
		}

		fo(i, j) = value;
	}
}

/**********INLINE FUNCTION DEFINITIONS**********/
inline int grid_size(int n, int threads)
{
	return ceil(float(n) / threads);
}
