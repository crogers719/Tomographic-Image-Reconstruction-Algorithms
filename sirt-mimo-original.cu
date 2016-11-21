// PROJECT: Ultrasonic tomographic imaging source code GPU
// Distribution is limited
// Date: December 15th, 2013
// University of Maryland Eastern Shore, Salisbury University, Florida International University
//
// AUTHORS: 
// Pedro D Bello-Maldonado, pbell005@fiu.edu, (786) 203-9025
// Yuanwei Jin, yjin@umes.edu, (410) 621-3410
//
// Copyright © 2013 All rights reserved

// Headers
#include <fstream>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Functions Declaration
void Ultrasonic_Tomography(int, float, int, int);

// Math functions
float norm(float*, int, int);
float norm(float*, float*, int, int);

// Utility Functions
int gridSize(int, int);
void save(const char*, int*, int);
void save(const char*, float*, int);
void save(const char*, float*, int, int);
void save(const char*, float*, int, int, int, int);

// Host Functions
void transducersPosition(int*&, int*&, int);

// Device Functions
__global__ void gridSetup(float*, float*);
__global__ void imagingField(float*, float*, float*);
__global__ void setSensorActivationVariables(float*, float*, float*, float*, float*);

__global__ void wavePropagation(float**, float*, float*, float*, float*, float, int, int*, int*, int*, int*);
__global__ void waveAtCorners(float**);
__global__ void waveAtSensors(float**, float*, float*, float*, float*, int);
__global__ void setAlpha(float*);
__global__ void differenceAtSensors(float**, float*, float*, float*, float*, float**, float**, float**, float**, int);
__global__ void setZ(float**);
__global__ void waveBackpropagation(float**, float*, float, int);
__global__ void waveBackpropagationBoundaryCorners(float**, float**, float**, float**, float**, int);
__global__ void deltau_k(float**, float**, int);
__global__ void df_k(float*, float**, float**, float*, int);
__global__ void f_k(float*, float*, float*);
__global__ void norm_f_f0_k(float*, float*, float*);
__global__ void norm_f0_k(float*, float*);

// Global Constants
#define Nx  201		//Number of sensors in  x and y
#define Ny  201
#define Nt  401		
#define Ns  640		
#define Ng  64 		// Number of sensor groups 640 / 10
#define Na  8  		// Number of groups to be allocated on the GPU for simultanous activation
#define Nm  40 		// Sensors pergroup
#define omega 16.0f

#define h   0.001f
#define hx  0.001f	//dimensions of pizel
#define hy  0.001f

#define pi 3.1415926535897932f

// CPU Consts
const float T_h = 0.2f / 1500.0f;
const float dt_h = T_h / 400.0f;
const float dt2_h = dt_h * dt_h;
const float fre_h = 125000.0f;
const float omegac_h = 2.0f * pi * fre_h;
const float tao_h = pi / omegac_h;
const float tao2_h = tao_h * tao_h;
const float tt_h = sqrt(6.0f * log(2.0f)) * tao_h;
const float h2_h = h * h;

const int Nx_Ny_h = Nx * Ny;
const int Nx_Ny_Nt_h = Nx_Ny_h * Nt;

const int Nt_Nx_h = Nt * Nx;
const int Nt_Ny_h = Nt * Ny;

const int blockDim_x = 16;
const int blockDim_y = 16;
const int nThreads = blockDim_x * blockDim_y;

/* const int t_sensors = 640; */
//const int t_iterations = 30;

// GPU Constants
__constant__ float T;
__constant__ float dt;
__constant__ float dt2;
__constant__ float fre;
__constant__ float omegac;
__constant__ float tao;
__constant__ float tao2;
__constant__ float tt;
__constant__ float h2;

__constant__ int Nx_Ny;
__constant__ int Nx_Ny_Nt;
__constant__ int Nt_Nx;
__constant__ int Nt_Ny;

int main(int argc, char** argv)
{
	
	if (argc != 4) {
		cerr << "Usage: " << argv[0] << " <sensor group size> <target epsilon> <max iterations>\n\n";
		exit(1);
	}

	int group_size = stoi(argv[1]);
	float target_epsilon = stof(argv[2]);
	int max_iterations = stoi(argv[3]);

	if (max_iterations == -1)
		max_iterations = numeric_limits<int>::max();
	
// Time measuring variables
	int t_start = 0, t_end = 0, t_total = 0;
	time_t clockTime;

	// Initial timings
	t_start = clock();
	time(&clockTime);

	// Start Imaging Process
	Ultrasonic_Tomography(group_size, target_epsilon, max_iterations, t_start);

	// Final timings
	time(&clockTime);
	t_end = clock();
	t_total = t_end - t_start;

	// Results
	printf("Clock Times:\n");
	/*printf("Initial Time: %s", ctime(&clockTime));
	printf("Final Time:   %s", ctime(&clockTime));*/

	printf("\nExecution Times:\n");
	printf("t_start = %d\n", t_start);
	printf("t_end   = %d\n", t_end);
	printf("t_total = %d\n", t_total);
	printf("Total Seconds = %f\n\n", (float)(t_total) / CLOCKS_PER_SEC);

	// End of the program
	return 0;
}

// Functions Definition
void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int t_start)
{
	// Iteration variables
	int i = 0, k = 0, p = 0;

	// Copy constant variables to GPU -- PROTOTYPE: cudaMemcpyToSymbol(Host, &Device, sizeof(datatype))
	cudaMemcpyToSymbol(T, &T_h, sizeof(float));
	cudaMemcpyToSymbol(dt, &dt_h, sizeof(float));
	cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(float));
	cudaMemcpyToSymbol(fre, &fre_h, sizeof(float));
	cudaMemcpyToSymbol(omegac, &omegac_h, sizeof(float));
	cudaMemcpyToSymbol(tao, &tao_h, sizeof(float));
	cudaMemcpyToSymbol(tao2, &tao2_h, sizeof(float));
	cudaMemcpyToSymbol(tt, &tt_h, sizeof(float));
	cudaMemcpyToSymbol(h2, &h2_h, sizeof(float));
	cudaMemcpyToSymbol(Nx_Ny, &Nx_Ny_h, sizeof(int));
	cudaMemcpyToSymbol(Nx_Ny_Nt, &Nx_Ny_Nt_h, sizeof(int));
	cudaMemcpyToSymbol(Nt_Nx, &Nt_Nx_h, sizeof(int));
	cudaMemcpyToSymbol(Nt_Ny, &Nt_Ny_h, sizeof(int));
	
	/*KERNEL PREPARATION*/
	int gridDim_x = gridSize(blockDim_x, Nx);		//gridSize returns (Nx+(blockDim_x -1))/blockDim_x
	int gridDim_y = gridSize(blockDim_y, Ny);
	//dim3 is an integer vector type that can be used in CUDA code - used to pass grid and block dimensions in a kernel invocation
	dim3 Block(blockDim_x, blockDim_y);		
	dim3 Grid(gridDim_x, gridDim_y);

	// Coordinates of the grid
	float* dev_x;
	int size_x = Nx * sizeof(float);
	cudaMalloc((void**)&dev_x, size_x);

	float* dev_y;
	int size_y = Ny * sizeof(float);
	cudaMalloc((void**)&dev_y, size_y);

	//call gridSetup device function
	gridSetup<<<dim3(gridDim_x, 1), Block>>>(dev_x, dev_y);

	//Imaging Field
	float* dev_f0;
	int size_f0 = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_f0, size_f0);
	cudaMemset(dev_f0, 0.0f, size_f0);

	imagingField<<<Grid, Block>>>(dev_f0, dev_x, dev_y);

	// Position of transducers
	int *ii, *jj;
	transducersPosition(ii, jj, Ns);

	// Sensor activation variables
	float* dev_v;
	int size_v = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_v, size_v);

	float* dev_r;
	int size_r = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_r, size_r);

	float* dev_r2;
	int size_r2 = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_r2, size_r2);

	float* dev_s;
	int size_s = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_s, size_s);

	setSensorActivationVariables<<<Grid, Block>>>(dev_f0, dev_v, dev_r, dev_r2, dev_s);

	// Wave propagation first scanning
	// Wave propagaing for each sensor group and corresponding transducers
	float* dev_u_ptr[Na];
	int size_u_ptr = Nx_Ny_Nt_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_u_ptr[i], size_u_ptr);
		cudaMemset(dev_u_ptr[i], 0.0f, size_u_ptr);
	}

	float **dev_u;
	int size_u = Na * sizeof(float*);
	cudaMalloc((void**)&dev_u, size_u);
	cudaMemcpy(dev_u, dev_u_ptr, size_u, cudaMemcpyHostToDevice);

	// Sensors
	int *dev_ip1;
	int size_ip1 = Na * sizeof(int);
	cudaMalloc((void**)&dev_ip1, size_ip1);

	int *dev_ip2;
	int size_ip2 = Na * sizeof(int);
	cudaMalloc((void**)&dev_ip2, size_ip2);

	int *dev_jp1;
	int size_jp1 = Na * sizeof(int);
	cudaMalloc((void**)&dev_jp1, size_jp1);

	int *dev_jp2;
	int size_jp2 = Na * sizeof(int);
	cudaMalloc((void**)&dev_jp2, size_jp2);

	int *ip1 = (int*)malloc(size_ip1);
	int *ip2 = (int*)malloc(size_ip2);
	int *jp1 = (int*)malloc(size_jp1);
	int *jp2 = (int*)malloc(size_jp2);	

	// Sensing variables for initial stage (REUSABLE)
	float* dev_g1;
	int size_g1 = (Nt_Nx_h * Ng) * sizeof(float);
	cudaMalloc((void**)&dev_g1, size_g1);
	cudaMemset(dev_g1, 0.0f, size_g1);

	float* dev_g3;
	int size_g3 = (Nt_Nx_h * Ng) * sizeof(float);
	cudaMalloc((void**)&dev_g3, size_g3);
	cudaMemset(dev_g3, 0.0f, size_g3);

	float* dev_g2;
	int size_g2 = (Nt_Nx_h * Ng) * sizeof(float);
	cudaMalloc((void**)&dev_g2, size_g2);
	cudaMemset(dev_g2, 0.0f, size_g2);

	float* dev_g4;
	int size_g4 = (Nt_Nx_h * Ng) * sizeof(float);
	cudaMalloc((void**)&dev_g4, size_g4);
	cudaMemset(dev_g4, 0.0f, size_g4);

	for (p = 0; p < Ns / (group_size * Na); p++)
	{	
		// Select the sensors to be used
		for (i = 0; i < Na; i++)
		{
			int jp1_h = jj[p * group_size * Na + i * group_size];
			int jp2_h = jj[p * group_size * Na + i * group_size + (group_size - 1)];
			int ip1_h = ii[p * group_size * Na + i * group_size];
			int ip2_h = ii[p * group_size * Na + i * group_size + (group_size - 1)];

			if (jp2_h < jp1_h)
			{
				int jp_h = jp1_h;
				jp1_h = jp2_h;
				jp2_h = jp_h;
			}

			if (ip2_h < ip1_h)
			{
				int ip_h = ip1_h;
				ip1_h = ip2_h;
				ip2_h = ip_h;
			}

			ip1[i] = ip1_h;
			ip2[i] = ip2_h;
			jp1[i] = jp1_h;
			jp2[i] = jp2_h;
		}

		cudaMemcpy(dev_ip1, ip1, size_ip1, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_ip2, ip2, size_ip2, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_jp1, jp1, size_jp1, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_jp2, jp2, size_jp2, cudaMemcpyHostToDevice);

		// Start scanning the imaging field
		// Boundary
		for (k = 1; k < Nt - 1; k++)
		{
			// Sensor activation
			float t = k * dt_h - tt_h;
			float wave = dt2_h * cos(omegac_h * t) * exp(-(t * t) / (2.0f * tao2_h));

			wavePropagation<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u, dev_v, dev_r, dev_r2, dev_s, wave, k, dev_ip1, dev_ip2, dev_jp1, dev_jp2);
		}

		// Corners
		waveAtCorners<<<dim3(gridSize(nThreads, Nt), 1, gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u);

		// Sensing
		waveAtSensors<<<dim3(gridSize(8, Nt), gridSize(8, 180), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u, dev_g1, dev_g3, dev_g2, dev_g4, p);
	}

	// Kaczmarz Method
	// Propagation
	float* dev_f;
	int size_f = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_f, size_f);
	cudaMemset(dev_f, 0.0f, size_f);
	
	// rr1
	float* dev_rr1_ptr[Na];
	int size_rr1_ptr = Nt_Nx_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_rr1_ptr[i], size_rr1_ptr);
		cudaMemset(dev_rr1_ptr[i], 0.0f, size_rr1_ptr);
	}

	float **dev_rr1;
	int size_rr1 = Na * sizeof(float*);
	cudaMalloc((void**)&dev_rr1, size_rr1);
	cudaMemcpy(dev_rr1, dev_rr1_ptr, size_rr1, cudaMemcpyHostToDevice);

	// rr3
	float* dev_rr3_ptr[Na];
	int size_rr3_ptr = Nt_Nx_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_rr3_ptr[i], size_rr3_ptr);
		cudaMemset(dev_rr3_ptr[i], 0.0f, size_rr3_ptr);
	}

	float **dev_rr3;
	int size_rr3 = Na * sizeof(float*);
	cudaMalloc((void**)&dev_rr3, size_rr3);
	cudaMemcpy(dev_rr3, dev_rr3_ptr, size_rr3, cudaMemcpyHostToDevice);

	// rr2
	float* dev_rr2_ptr[Na];
	int size_rr2_ptr = Nt_Ny_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_rr2_ptr[i], size_rr2_ptr);
		cudaMemset(dev_rr2_ptr[i], 0.0f, size_rr2_ptr);
	}

	float **dev_rr2;
	int size_rr2 = Na * sizeof(float*);
	cudaMalloc((void**)&dev_rr2, size_rr2);
	cudaMemcpy(dev_rr2, dev_rr2_ptr, size_rr2, cudaMemcpyHostToDevice);

	// rr4
	float* dev_rr4_ptr[Na];
	int size_rr4_ptr = Nt_Ny_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_rr4_ptr[i], size_rr4_ptr);
		cudaMemset(dev_rr4_ptr[i], 0.0f, size_rr4_ptr);
	}

	float **dev_rr4;
	int size_rr4 = Na * sizeof(float*);
	cudaMalloc((void**)&dev_rr4, size_rr4);
	cudaMemcpy(dev_rr4, dev_rr4_ptr, size_rr4, cudaMemcpyHostToDevice);

	// Backpropagation Variables
	float* dev_z_ptr[Na];
	int size_z_ptr = (Nx_Ny_h * (Nt + 1)) * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_z_ptr[i], size_z_ptr);
		cudaMemset(dev_z_ptr[i], 0.0f, size_z_ptr);
	}

	float **dev_z;
	int size_z = Na * sizeof(float*);
	cudaMalloc((void**)&dev_z, size_z);
	cudaMemcpy(dev_z, dev_z_ptr, size_z, cudaMemcpyHostToDevice);

	float* dev_Lu_ptr[Na];
	int size_Lu_ptr = Nx_Ny_Nt_h * sizeof(float);

	for (i = 0; i < Na; i++)
	{
		cudaMalloc((void**)&dev_Lu_ptr[i], size_Lu_ptr);
		cudaMemset(dev_Lu_ptr[i], 0.0f, size_Lu_ptr);
	}

	float **dev_Lu;
	int size_Lu = Na * sizeof(float*);
	cudaMalloc((void**)&dev_Lu, size_Lu);
	cudaMemcpy(dev_Lu, dev_Lu_ptr, size_Lu, cudaMemcpyHostToDevice);

	float* dev_df;
	int size_df = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_df, size_df);

	float* dev_alpha;
	int size_alpha = Nx_Ny_h * sizeof(float);
	cudaMalloc((void**)&dev_alpha, size_alpha);
	cudaMemset(dev_alpha, 0.0f, size_alpha);
	setAlpha<<<dim3(gridSize(blockDim_x, 159), gridSize(blockDim_y, 159)), Block>>>(dev_alpha);

	float* dev_norm_f0;
	int size_norm_f0 = sizeof(float);
	cudaMalloc((void**)&dev_norm_f0, size_norm_f0);
	cudaMemset(dev_norm_f0, 0.0f, size_norm_f0);

	float* dev_norm_f_f0;
	int size_norm_f_f0 = sizeof(float);
	cudaMalloc((void**)&dev_norm_f_f0, size_norm_f_f0);
	cudaMemset(dev_norm_f_f0, 0.0f, size_norm_f_f0);

	float* f = new float[Nx_Ny_h];
	float* f0 = new float[Nx_Ny_h];

	// Save convergence
	ofstream convergence_file;
	convergence_file.open("sirt_convergence.txt");
	ofstream time_file;
	time_file.open ("sirt_time.txt");

	for (int iter = 0; iter < max_iterations; iter++)
	{
		printf("iter: %d\n", iter);
		float epsilon = 0.0;

		// Initialize u for new sensor activation
		for (i = 0; i < Na; i++)
		{
			cudaMemset(dev_u_ptr[i], 0.0f, size_u_ptr);
		}

		setSensorActivationVariables<<<Grid, Block>>>(dev_f, dev_v, dev_r, dev_r2, dev_s);

		cudaMemset(dev_df, 0.0f, size_df);

		for (p = 0; p < Ns / (group_size * Na); p++)
		{
			// Select the sensors to be used
			for (i = 0; i < Na; i++)
			{
				int jp1_h = jj[p * group_size * Na + i * group_size];
				int jp2_h = jj[p * group_size * Na + i * group_size + (group_size - 1)];
				int ip1_h = ii[p * group_size * Na + i * group_size];
				int ip2_h = ii[p * group_size * Na + i * group_size + (group_size - 1)];

				if (jp2_h < jp1_h)
				{
					int jp_h = jp1_h;
					jp1_h = jp2_h;
					jp2_h = jp_h;
				}

				if (ip2_h < ip1_h)
				{
					int ip_h = ip1_h;
					ip1_h = ip2_h;
					ip2_h = ip_h;
				}

				ip1[i] = ip1_h;
				ip2[i] = ip2_h;
				jp1[i] = jp1_h;
				jp2[i] = jp2_h;
			}

			cudaMemcpy(dev_ip1, ip1, size_ip1, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_ip2, ip2, size_ip2, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_jp1, jp1, size_jp1, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_jp2, jp2, size_jp2, cudaMemcpyHostToDevice);

			// Start scanning the imaging field
			// Boundary
			for (k = 1; k < Nt - 1; k++)
			{
				// Sensor activation
				float t = k * dt_h - tt_h;
				float wave = dt2_h * cos(omegac_h * t) * exp(-(t * t) / (2.0f * tao2_h));

				wavePropagation<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u, dev_v, dev_r, dev_r2, dev_s, wave, k, dev_ip1, dev_ip2, dev_jp1, dev_jp2);
			}

			// Corners
			waveAtCorners<<<dim3(gridSize(nThreads, Nt), 1, gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u);

			// Sensing
			differenceAtSensors<<<dim3(gridSize(8, Nt), gridSize(8, 180), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_u, dev_g1, dev_g3, dev_g2, dev_g4, dev_rr1, dev_rr3, dev_rr2, dev_rr4, p);

			// Backpropagation
			setZ<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_z);

			float cons = 1500.0f * 1500.0f * dt2_h;

			for (k = Nt - 2; k >= 1; k--)
			{
				waveBackpropagation<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Na)), dim3(8, 8, 4)>>>(dev_z, dev_f, cons, k);
				waveBackpropagationBoundaryCorners<<<dim3(gridSize(8 * 8, Nx), 1, gridSize(4, Na)), dim3(8, 8, 4)>>> (dev_z, dev_rr1, dev_rr3, dev_rr2, dev_rr4, k);
			}

			// Deltau Calculation
			for (int pm = 0; pm < Na; pm++)
			{
				deltau_k<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Nt)), dim3(8, 8, 4)>>>(dev_Lu, dev_u, pm);

				df_k<<<dim3(gridSize(8, Nx), gridSize(8, Ny), gridSize(4, Nt)), dim3(8, 8, 4)>>>(dev_df, dev_z, dev_Lu, dev_f, pm);
			}
		}

		f_k<<<Grid, Block>>>(dev_f, dev_alpha, dev_df);

		cudaMemcpy(f, dev_f, size_f, cudaMemcpyDeviceToHost);
		cudaMemcpy(f0, dev_f0, size_f0, cudaMemcpyDeviceToHost);

		epsilon = norm(f, f0, Nx, Ny) / norm(f0, Nx, Ny) * 100.0f;
		printf("epsilon (%d): %f\n", p, epsilon);
		float current_t = (float)(clock()-t_start) / CLOCKS_PER_SEC;

		convergence_file << epsilon;
		convergence_file << " ";
		
		time_file << (current_t);
		time_file<<" ";

		if (epsilon <= target_epsilon)
		{
			break;
		}
	}

	// Save results
	//save("dev_x.txt", dev_x, Nx);
	//save("dev_y.txt", dev_y, Ny);
	save("dev_f0.txt", dev_f0, Nx, Ny);					//output files(image)
	//save("dev_fb.txt", dev_fb, Nx, Ny);
	//save("dev_ib.txt", dev_ib, Ibase[0]);
	//save("dev_jb.txt", dev_jb, Ibase[0]);
	//save("dev_v.txt", dev_v, Nx, Ny);
	//save("dev_s.txt", dev_s, Nx, Ny);
	//save("dev_u.txt", dev_u_ptr[Na - 1], Nx, Ny, Nt, Nt - 1);
	//save("dev_u.txt", dev_u_ptr[0], Nx, Ny, Nt, Nt - 1);
	//save("dev_u.txt", dev_u, Nx, Ny, Nt, Nt - 1);
	//save("dev_g1.txt", dev_g1, Nt, 180, Ng, 0);
	//save("dev_g4.txt", dev_g4, Nt, 180, Ns, 0);
	//save("dev_rr1.txt", dev_rr1_ptr[0], Nt, 180);
	//save("dev_rr4.txt", dev_rr4, Nt, 180);
	//save("dev_z.txt", dev_z_ptr[0], Nx, Ny, Nt + 1, 1);
	//save("dev_Lu.txt", dev_Lu_ptr[Na - 1], Nx, Ny, Nt, Nt - 1);
	//save("dev_df.txt", dev_df, Nx, Ny);
	//save("dev_alpha.txt", dev_alpha, Nx, Ny);
	save("dev_f.txt", dev_f, Nx, Ny);
	//save("dev_xf.txt", dev_xf, Ibase);

	// Release Memory
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_f0);
	delete[] ii;
	delete[] jj;
	cudaFree(dev_v);
	cudaFree(dev_r);
	cudaFree(dev_r2);
	cudaFree(dev_s);
	cudaFree(dev_u);
	cudaFree(dev_g1);
	cudaFree(dev_g3);
	cudaFree(dev_g2);
	cudaFree(dev_g4);
	cudaFree(dev_f);
	cudaFree(dev_rr1);
	cudaFree(dev_rr3);
	cudaFree(dev_rr2);
	cudaFree(dev_rr4);
	cudaFree(dev_z);
	cudaFree(dev_Lu);
	cudaFree(dev_df);
	cudaFree(dev_norm_f0);
	cudaFree(dev_norm_f_f0);
	delete[] f;
	delete[] f0;
	
	cudaFree(dev_u);
	cudaFree(dev_ip1);
	cudaFree(dev_ip2);
	cudaFree(dev_jp1);
	cudaFree(dev_jp2);
	delete[] ip1;
	delete[] ip2;
	delete[] jp1;
	delete[] jp2;
}

// Host Functions
void transducersPosition(int *&ii, int *&jj, int num)
{
	/*
	Returns the (x, y) coordinates of each of the 'num' total transducers.
	*/
	// Position of transducers
	ii = (int*)malloc(num * sizeof(int));
	jj = (int*)malloc(num * sizeof(int));

	int p = 0;

	for (p = 0; p < 160; p++)
	{
		ii[p] = 20 + (p + 1);
		jj[p] = 180;
	}

	for (p = 160; p < 320; p++)
	{
		ii[p] = 180;
		jj[p] = 180 - ((p + 1) - 160);
	}

	for (p = 320; p < 480; p++)
	{
		ii[p] = 180 - ((p + 1) - 320);
		jj[p] = 20;
	}

	for (p = 480; p < 640; p++)
	{
		ii[p] = 20;
		jj[p] = 20 + ((p + 1) - 480);
	}
}

// Device Functions
__global__ void gridSetup(float* x, float* y)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int index = i + j * blockDim.x * gridDim.x;

	if (index < Nx)
	{
		float div = -0.1f + (float)(index)* h;
		x[index] = div;
		y[index] = div;
	}
}

__global__ void imagingField(float* f0, float* x, float* y)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < Nx) && (j < Ny))
	{
		int index = i + j * Nx;

		// Ground truth image
		/*if (sqrt(powf(x[i] - 0.04f, 2) + powf(y[j] - 0.04f, 2)) <= 0.003f)
		{
		f0[index] = 0.06f;
		}
		else if (sqrt(powf(x[i], 2) + powf(y[j], 2)) <= 0.005)
		{
		f0[index] = 0.06f;
		}
		else if (sqrt(powf(x[i] + 0.04f, 2) + powf(y[j] + 0.04f, 2)) <= 0.004f)
		{
		f0[index] = 0.06f;
		}*/

		/*if(((sqrtf(powf(x[i] - 0.015f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.015f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f))
		{
		f0[index] = 0.06f;
		}
		else
		{
		if(sqrtf(x[i] * x[i] + y[j] * y[j]) <= 0.03f)
		{
		f0[index] = 0.02f;
		}
		else
		{
		f0[index] = 0.0f;
		}
		}*/

		float rc = 0.015f;
		float rp = 0.005f;

		float sc = 0.03f;
		float sp = 0.05f;

		if (powf(x[i], 2) + powf(y[j], 2) <= powf(rc, 2))
		{
			f0[index] = sc;
		}

		if (powf(x[i] - rc * cos(-30 * (pi / 180)), 2) + powf(y[j] - rc * sin(30 * (pi / 180)), 2) <= powf(rp, 2))
		{
			f0[index] = sp;
		}

		if (powf(x[i] + rc * cos(-30 * (pi / 180)), 2) + powf(y[j] - rc * sin(30 * (pi / 180)), 2) <= powf(rp, 2))
		{
			f0[index] = sp;
		}

		if (powf(x[i], 2) + powf(y[j] + rc, 2) <= powf(rp, 2))
		{
			f0[index] = sp;
		}
	}
}

__global__ void setSensorActivationVariables(float* f0, float* v, float* r, float* r2, float* s)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int index = i + j * blockDim.x * gridDim.x;

	if (index < Nx_Ny)
	{
		//Propagation speed of acoustical wave in the medium : 	c(x)=c0*sqrt(1+f(x))
		//(c0 = ambient sound speed ; f(x) = the acoustical potential function that needs to be reconstructed)
		float v_c = 1500.0f * sqrt(1.0f + f0[index]);
		
		float r_c = v_c * dt / h;
		float r2_c = r_c * r_c;

		v[index] = v_c;
		r[index] = r_c;
		r2[index] = r2_c;
		s[index] = 2.0f - 4.0f * r2_c;
	}
}

__global__ void wavePropagation(float **u, float *v, float *r, float *r2, float *s, float wave, int k, int *ip1, int *ip2, int *jp1, int *jp2)
{
	// Thread Mapping
	int i = (threadIdx.x + blockIdx.x * blockDim.x) + 1;
	int j = (threadIdx.y + blockIdx.y * blockDim.y) + 1;
	int p = threadIdx.z + blockIdx.z * blockDim.z;

	if ((i < (Nx - 1)) && (j < (Ny - 1)) && (p < Na))
	{
		int index = i + j * Nx;

		int k_Nx_Ny = k * Nx_Ny;
		int k_p_Nx_Ny = k_Nx_Ny + Nx_Ny;
		int k_n_Nx_Ny = k_Nx_Ny - Nx_Ny;
		int index_k_Nx_Ny = index + k_Nx_Ny;

		if ((j >= jp1[p]) && (j <= jp2[p]) && (i >= ip1[p]) && (i <= ip2[p]) && (k < 24))
		{
			u[p][index + k_p_Nx_Ny] = v[index] * v[index] * wave + r2[index] * (u[p][index_k_Nx_Ny + 1] + u[p][index_k_Nx_Ny - 1] + u[p][index_k_Nx_Ny + Nx] + u[p][index_k_Nx_Ny - Nx]) + s[index] * u[p][index_k_Nx_Ny] - u[p][index + k_n_Nx_Ny];
		}
		else
		{
			u[p][index + k_p_Nx_Ny] = r2[index] * (u[p][index_k_Nx_Ny + 1] + u[p][index_k_Nx_Ny - 1] + u[p][index_k_Nx_Ny + Nx] + u[p][index_k_Nx_Ny - Nx]) + s[index] * u[p][index_k_Nx_Ny] - u[p][index + k_n_Nx_Ny];
		}

		if (j == 1)
		{
			// Right boundary
			int index_i = i * Nx;
			index_k_Nx_Ny = index_i + k_Nx_Ny;
			float two_r = 2.0f * r[index_i];
			u[p][index_i + k_p_Nx_Ny] = (2.0f - two_r - r2[index_i]) * u[p][index_k_Nx_Ny] + two_r * (1.0f + r[index_i]) * u[p][index_k_Nx_Ny + 1] - r2[index_i] * u[p][index_k_Nx_Ny + 2] + (two_r - 1.0f) * u[p][index_i + k_n_Nx_Ny] - two_r * u[p][index_i + 1 + k_n_Nx_Ny];

			// Left boundary
			index_i = (Nx - 1) + index_i;
			index_k_Nx_Ny = index_i + k_Nx_Ny;
			two_r = 2.0f * r[index_i];
			u[p][index_i + k_p_Nx_Ny] = (2.0f - two_r - r2[index_i]) * u[p][index_k_Nx_Ny] + two_r * (1.0f + r[index_i]) * u[p][index_k_Nx_Ny - 1] - r2[index_i] * u[p][index_k_Nx_Ny - 2] + (two_r - 1.0f) * u[p][index_i + k_n_Nx_Ny] - two_r * u[p][index_i - 1 + k_n_Nx_Ny];

			// Upper boundary
			index_i = i;
			index_k_Nx_Ny = index_i + k_Nx_Ny;
			two_r = 2.0f * r[index_i];
			u[p][index_i + k_p_Nx_Ny] = (2.0f - two_r - r2[index_i]) * u[p][index_k_Nx_Ny] + two_r * (1.0f + r[index_i]) * u[p][index_k_Nx_Ny + Nx] - r2[index_i] * u[p][index_k_Nx_Ny + Nx + Nx] + (two_r - 1.0f) * u[p][index_i + k_n_Nx_Ny] - two_r * u[p][index_i + Nx + k_n_Nx_Ny];

			// Lower boundary
			index_i = index_i + (Nx_Ny - Nx);
			index_k_Nx_Ny = index_i + k_Nx_Ny;
			two_r = 2.0f * r[index_i];
			u[p][index_i + k_p_Nx_Ny] = (2.0f - two_r - r2[index_i]) * u[p][index_k_Nx_Ny] + two_r * (1.0f + r[index_i]) * u[p][index_k_Nx_Ny - Nx] - r2[index_i] * u[p][index_k_Nx_Ny - Nx - Nx] + (two_r - 1.0f) * u[p][index_i + k_n_Nx_Ny] - two_r * u[p][index_i - Nx + k_n_Nx_Ny];
		}
	}
}

__global__ void waveAtCorners(float **u)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int p = threadIdx.z + blockIdx.z * blockDim.z;
	int k = i + j * blockDim.x * gridDim.x;

	if ((k < Nt) && (p < Na))
	{
		int k_Nx_Ny = k * Nx_Ny;
		int Nx_Ny_1 = Nx_Ny - Nx;

		u[p][k_Nx_Ny] = 0.5f * (u[p][1 + k_Nx_Ny] + u[p][Nx + k_Nx_Ny]);
		u[p][(Nx - 1) + k_Nx_Ny] = 0.5f * (u[p][(Nx - 2) + k_Nx_Ny] + u[p][Nx - 1 + Nx + k_Nx_Ny]);
		u[p][Nx_Ny_1 + k_Nx_Ny] = 0.5f * (u[p][Nx_Ny_1 - Nx + k_Nx_Ny] + u[p][1 + Nx_Ny_1 + k_Nx_Ny]);
		u[p][(Nx - 1) + Nx_Ny_1 + k_Nx_Ny] = 0.5f * (u[p][(Nx - 2) + Nx_Ny_1 + k_Nx_Ny] + u[p][Nx_Ny_1 - 1 + k_Nx_Ny]);
	}
}

__global__ void waveAtSensors(float **u, float *g1, float *g3, float *g2, float *g4, int pm)
{
	// Thread Mapping
	int k = threadIdx.x + blockIdx.x * blockDim.x + 2;
	int i = threadIdx.y + blockIdx.y * blockDim.y + 21;
	int p = threadIdx.z + blockIdx.z * blockDim.z;

	if ((k < Nt) && (i < 180) && (p < Na))
	{
		int p_Nt_Nx = (p + pm * Na) * Nt_Nx;

		int index = k + i * Nt + p_Nt_Nx;
		g1[index] = u[p][i + 180 * Nx + (k * Nx_Ny)];
		g3[index] = u[p][i + 20 * Nx + (k * Nx_Ny)];

		index = k + i * Nt + p_Nt_Nx;
		g2[index] = u[p][180 + (i*Nx) + (k * Nx_Ny)];
		g4[index] = u[p][20 + (i*Nx) + (k * Nx_Ny)];
	}
}

__global__ void setAlpha(float* alpha)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x + 21;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 21;

	if ((i < 180) && (j < 180))
	{
		int index = i + j * Nx;

		alpha[index] = 1.0f;
	}
}

__global__ void differenceAtSensors(float **u, float *g1, float *g3, float *g2, float *g4, float **rr1, float **rr3, float **rr2, float **rr4, int pm)
{
	// Thread Mapping
	int k = threadIdx.x + blockIdx.x * blockDim.x + 2;
	int i = threadIdx.y + blockIdx.y * blockDim.y + 21;
	int p = threadIdx.z + blockIdx.z * blockDim.z;
	
	if ((k < Nt) && (i < 180) && (p < Na))
	{
		int p_Nt_Nx = (p + pm * Na) * Nt_Nx;
		int k_Nx_Ny = k * Nx_Ny;

		int index_rr = k + i * Nt;
		int index_g = index_rr + p_Nt_Nx;

		rr1[p][index_rr] = g1[index_g] - u[p][i + 180 * Nx + k_Nx_Ny];
		rr3[p][index_rr] = g3[index_g] - u[p][i + 20 * Nx + k_Nx_Ny];

		index_rr = k + i * Nt;
		index_g = index_rr + p_Nt_Nx;
		int i_Nx = i * Nx;

		rr2[p][index_rr] = g2[index_g] - u[p][180 + i_Nx + k_Nx_Ny];
		rr4[p][index_rr] = g4[index_g] - u[p][20 + i_Nx + k_Nx_Ny];
	}
}

__global__ void setZ(float **z)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int p = threadIdx.z + blockIdx.z * blockDim.z;

	if ((i < Nx) && (j < Ny) && (p < Na))
	{
		int index_Nx_Ny_Nt = i + j * Nx + Nx_Ny_Nt;

		z[p][index_Nx_Ny_Nt - Nx_Ny] = 0.0f;
		z[p][index_Nx_Ny_Nt] = 0.0f;
	}
}

__global__ void waveBackpropagation(float **z, float *f, float cons, int k)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int p = threadIdx.z + blockIdx.z * blockDim.z;

	//setting boundaries
	if ((i < (Nx - 1)) && (j < (Ny - 1)) && (p < Na))
	{
		int index = i + j * Nx;

		int k_Nx_Ny = k * Nx_Ny;
		int k_p_Nx_Ny = k_Nx_Ny + Nx_Ny;
		int index_k_p_Nx_Ny = index + k_p_Nx_Ny;

		z[p][index + k_Nx_Ny] = cons * ((1.0f + f[index - Nx]) * z[p][index_k_p_Nx_Ny - Nx] + (1.0f + f[index + Nx]) * z[p][index_k_p_Nx_Ny + Nx] + (1.0f + f[index - 1]) * z[p][index_k_p_Nx_Ny - 1] + (1.0f + f[index + 1]) * z[p][index_k_p_Nx_Ny + 1] - 4.0f * (1.0f + f[index]) * z[p][index_k_p_Nx_Ny]) / h2 + 2.0f * z[p][index_k_p_Nx_Ny] - z[p][index_k_p_Nx_Ny + Nx_Ny];
	}
}

__global__ void waveBackpropagationBoundaryCorners(float **z, float **rr1, float **rr3, float **rr2, float **rr4, int k)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y;
	int index = i + j * blockDim.x * gridDim.x;
	int p = threadIdx.z + blockIdx.z * blockDim.z;
	
	//four corners need to be handled seperately--difference signals....

	if ((index < (Nx - 1)) && (p < Na))
	{
		int k_Nx_Ny = k * Nx_Ny;
		int j_Nx_k_Nx_Ny = index * Nx + k_Nx_Ny;

		z[p][index + k_Nx_Ny] = z[p][index + Nx + k_Nx_Ny];
		z[p][index + (Nx_Ny - Nx) + k_Nx_Ny] = z[p][index + (Nx_Ny - Nx - Nx) + k_Nx_Ny];

		z[p][j_Nx_k_Nx_Ny] = z[p][1 + j_Nx_k_Nx_Ny];
		z[p][(Nx - 1) + j_Nx_k_Nx_Ny] = z[p][(Nx - 2) + j_Nx_k_Nx_Ny];

		if ((index >= 21) && (index < 180))
		{
			z[p][index + 180 * Nx + k_Nx_Ny] = z[p][index + 179 * Nx + k_Nx_Ny] + rr1[p][k + index * Nt];
			z[p][index + 20 * Nx + k_Nx_Ny] = z[p][index + 21 * Nx + k_Nx_Ny] + rr3[p][k + index * Nt];

			int j_Nt = index * Nt;

			z[p][180 + j_Nx_k_Nx_Ny] = z[p][179 + j_Nx_k_Nx_Ny] + rr2[p][k + j_Nt];
			z[p][20 + j_Nx_k_Nx_Ny] = z[p][21 + j_Nx_k_Nx_Ny] + rr4[p][k + j_Nt];
		}

		z[p][k_Nx_Ny] = (z[p][1 + k_Nx_Ny] + z[p][Nx + k_Nx_Ny]) / 2.0f;
		z[p][(Nx - 1) + k_Nx_Ny] = (z[p][(Nx - 2) + k_Nx_Ny] + z[p][(Nx - 1) + Nx + k_Nx_Ny]) / 2.0f;
		z[p][(Nx_Ny - Nx) + k_Nx_Ny] = (z[p][1 + (Nx_Ny - Nx) + k_Nx_Ny] + z[p][(Nx_Ny - Nx - Nx) + k_Nx_Ny]) / 2.0f;
		z[p][(Nx - 1) + (Nx_Ny - Nx) + k_Nx_Ny] = (z[p][(Nx - 2) + (Nx_Ny - Nx) + k_Nx_Ny] + z[p][(Nx - 1) + (Nx_Ny - Nx - Nx) + k_Nx_Ny]) / 2.0f;
	}
}

__global__ void deltau_k(float **Lu, float **u, int p)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if ((i < (Nx - 1)) && (j < (Ny - 1)) && (k < Nt))
	{
		int index = i + j * Nx;
		int k_Nx_Ny = k * Nx_Ny;
		int index_k_Nx_Ny = index + k_Nx_Ny;
		
		//Lu =delta u
		Lu[p][index_k_Nx_Ny] = (u[p][index_k_Nx_Ny - Nx] + u[p][index_k_Nx_Ny + Nx] + u[p][index_k_Nx_Ny - 1] + u[p][index_k_Nx_Ny + 1] - 4.0f * u[p][index_k_Nx_Ny]) / h2;

		if (j == 1)
		{
			index = i;
			index_k_Nx_Ny = index + k_Nx_Ny;
			int index_j_k_Nx_Ny = index_k_Nx_Ny + (Nx_Ny - Nx);

			Lu[p][index_k_Nx_Ny] = (u[p][index_k_Nx_Ny] + u[p][index_k_Nx_Ny + Nx] + u[p][index_k_Nx_Ny - 1] + u[p][index_k_Nx_Ny + 1] - 4.0f * u[p][index_k_Nx_Ny]) / h2;
			Lu[p][index_j_k_Nx_Ny] = (u[p][index_j_k_Nx_Ny] + u[p][index_j_k_Nx_Ny - Nx] + u[p][index_j_k_Nx_Ny - 1] + u[p][index_j_k_Nx_Ny + 1] - 4.0f * u[p][index_j_k_Nx_Ny]) / h2;

			index = i * Nx;
			index_k_Nx_Ny = index + k_Nx_Ny;
			int index_i_k_Nx_Ny = index_k_Nx_Ny + (Nx - 1);

			Lu[p][index_k_Nx_Ny] = (u[p][index_k_Nx_Ny] + u[p][index_k_Nx_Ny + 1] + u[p][index_k_Nx_Ny - Nx] + u[p][index_k_Nx_Ny + Nx] - 4.0f * u[p][index_k_Nx_Ny]) / h2;
			Lu[p][index_i_k_Nx_Ny] = (u[p][index_i_k_Nx_Ny] + u[p][index_i_k_Nx_Ny - 1] + u[p][index_i_k_Nx_Ny - Nx] + u[p][index_i_k_Nx_Ny + Nx] - 4.0f * u[p][index_i_k_Nx_Ny]) / h2;

			Lu[p][k_Nx_Ny] = (Lu[p][1 + k_Nx_Ny] + Lu[p][Nx + k_Nx_Ny]) / 2.0f;
			Lu[p][(Nx - 1) + k_Nx_Ny] = (Lu[p][(Nx - 2) + k_Nx_Ny] + Lu[p][(Nx - 1) + Nx + k_Nx_Ny]) / 2.0f;
			Lu[p][(Nx_Ny - Nx) + k_Nx_Ny] = (Lu[p][1 + (Nx_Ny - Nx) + k_Nx_Ny] + Lu[p][(Nx_Ny - Nx - Nx) + k_Nx_Ny]) / 2.0f;
			Lu[p][(Nx - 1) + (Nx_Ny - Nx) + k_Nx_Ny] = (Lu[p][(Nx - 2) + (Nx_Ny - Nx) + k_Nx_Ny] + Lu[p][(Nx - 1) + (Nx_Ny - Nx - Nx) + k_Nx_Ny]) / 2.0f;
		}
	}
}

__global__ void df_k(float *df, float **z, float **Lu, float *f, int p)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if ((i < (Nx - 1)) && (j < (Ny - 1)) && (k < Nt))
	{
		int index = i + j * Nx;
		int k_Nx_Ny = k * Nx_Ny;
		int index_k_Nx_Ny = index + k_Nx_Ny;

		atomicAdd(&(df[index]), z[p][index_k_Nx_Ny] * Lu[p][index_k_Nx_Ny] / (1.0f + f[index]));
	}
}

__global__ void f_k(float* f, float* alpha, float* df)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < Nx) && (j < Ny))
	{
		int index = i + j * Nx;

		f[index] += 20000.0f * alpha[index] * df[index] / omega;
	}
}

__global__ void norm_f_f0_k(float* f, float* f0, float* norm_f_f0)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y;
	int index = i + j * blockDim.x * gridDim.x;

	__shared__ float sum_c[nThreads];
	int index_c = threadIdx.x + threadIdx.y * blockDim.x;
	float sum = 0.0f;

	while (index < Nx_Ny)
	{
		float f_minus_f0 = f[index] - f0[index];
		sum += f_minus_f0 * f_minus_f0;
		index += gridDim.x * gridDim.y * nThreads;
	}

	sum_c[index_c] = sum;

	__syncthreads();

	int div = nThreads / 2;

	while (div != 0)
	{
		if (index_c < div)
		{
			sum_c[index_c] += sum_c[index_c + div];
		}

		__syncthreads();
		div /= 2;
	}

	if (index_c == 0)
	{
		atomicAdd(norm_f_f0, sum_c[0]);
	}
}

__global__ void norm_f0_k(float* f, float* norm_f)
{
	// Thread Mapping
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y;
	int index = i + j * blockDim.x * gridDim.x;

	__shared__ float sum_c[nThreads];
	int index_c = threadIdx.x + threadIdx.y * blockDim.x;
	float sum = 0.0f;

	while (index < Nx_Ny)
	{
		float f_i = f[index];
		sum += f_i * f_i;
		index += gridDim.x * gridDim.y * nThreads;
	}

	sum_c[index_c] = sum;

	__syncthreads();

	int div = nThreads / 2;

	while (div != 0)
	{
		if (index_c < div)
		{
			sum_c[index_c] += sum_c[index_c + div];
		}

		__syncthreads();
		div /= 2;
	}

	if (index_c == 0)
	{
		atomicAdd(norm_f, sum_c[0]);
	}
}

// Math functions
float norm(float* A, int n, int m)
{
	int i = 0, j = 0;
	float sum = 0.0f;

	for (j = 0; j < m; j++)
	{
		int j_n = j * n;

		for (i = 0; i < n; i++)
		{
			int index = i + j_n;

			sum += A[index] * A[index];
		}
	}

	return sqrtf(sum);
}

float norm(float* A, float* B, int n, int m)
{
	int i = 0, j = 0;
	float sum = 0.0f;

	for (j = 0; j < m; j++)
	{
		int j_n = j * n;

		for (i = 0; i < n; i++)
		{
			int index = i + j_n;

			float A_minus_B = A[index] - B[index];
			sum += A_minus_B * A_minus_B;
		}
	}

	return sqrtf(sum);
}

// Utility functions
int gridSize(int threads, int N)
{
	return (N + (threads - 1)) / threads;
}

void save(const char* name, int* arr, int n)
{
	// Create file
	ofstream file;
	file.open(name);

	// Create temporal points
	int* arr_t = new int[n];

	// Copy array from GPU to CPU
	cudaMemcpy(arr_t, arr, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Copy data
	for (int i = 0; i < n; i++)
	{
		file << arr_t[i];
		file << " ";
	}

	// Close and release the file
	file.close();

	// Delete temporary pointer
	delete(arr_t);
}

void save(const char* name, float* arr, int n)
{
	// Create file
	ofstream file;
	file.open(name);

	// Create temporal points
	float* arr_t = new float[n];

	// Copy array from GPU to CPU
	cudaMemcpy(arr_t, arr, n * sizeof(float), cudaMemcpyDeviceToHost);

	// Copy data
	for (int i = 0; i < n; i++)
	{
		file << arr_t[i];
		file << " ";
	}

	// Close and release the file
	file.close();

	// Delete temporary pointer
	delete(arr_t);
}

void save(const char* name, float* arr, int n, int m)		//only need this save??
{
	// Create file
	ofstream file;
	file.open(name);

	// Create temporal points
	float* arr_t = new float[n * m];

	// Copy array from GPU to CPU
	cudaMemcpy(arr_t, arr, (n * m) * sizeof(float), cudaMemcpyDeviceToHost);

	// Copy data
	for (int j = 0; j < m; j++)
	{
		int j_n = j * n;

		for (int i = 0; i < n; i++)
		{
			file << arr_t[i + j_n];
			file << " ";
		}

		file << "\n";
	}

	// Close and release the file
	file.close();

	// Delete temporary pointer
	delete(arr_t);
}

void save(const char* name, float* arr, int n, int m, int o, int k)
{
	// Create file
	ofstream file;
	file.open(name);

	// Create temporal points
	float* arr_t = new float[n * m * o];

	// Copy array from GPU to CPU
	cudaMemcpy(arr_t, arr, (n * m * o) * sizeof(float), cudaMemcpyDeviceToHost);

	// Copy data
	int offset = k * n * m;

	for (int j = 0; j < m; j++)
	{
		int j_n = j * n;

		for (int i = 0; i < n; i++)
		{
			file << arr_t[i + j_n + offset];
			file << " ";
		}

		file << "\n";
	}

	// Close and release the file
	file.close();

	// Delete temporary pointer
	delete(arr_t);
}
