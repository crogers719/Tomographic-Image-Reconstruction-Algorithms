// HEADERS

#include <iostream>
#include <iomanip>
#include <limits>

#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;

// DEFINITIONS

#define NX		201
#define NY		201
#define NT		401
#define NS		640


__constant__ float hx = 0.001f;
__constant__ float hy = 0.001f;
__constant__ float h = 0.001f;
__constant__ float T = 1.3333e-04f; // 0.2f / 1500.0f;
__constant__ float dt = 3.3333e-07f; // T / 400.0f;
__constant__ float fre = 125000.0f;
__constant__ float omegac = 7.8540e+05f; // 2.0f * pi * fre;
__constant__ float tao = 4.0000e-06f; // pi / omegac;
__constant__ float tt = 8.1573e-06f; // sqrtf(6.0f * logf(2.0f)) * tao;

// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti);
void IO_Files(float*, float*, float*, float*);
float norm(float*, int);

__global__ void field_setup(float*, float*, float*, float*, float*, float*, float*);
__global__ void propagation(int, int, int, int, float*, float*, float* , float*, float*, int);
__global__ void propagation_at_corners(float*);
__global__ void initial_signal(float*, float*, float*, float*, float*, int);
__global__ void difference_signal(float*, float*, float*, float*, float*, float*, float*, float*, float*, int);
__global__ void backpropagation1(float*, float*, int);
__global__ void backpropagation2(float*, float*, float*, float*, float*, int);
__global__ void laplace1(float*, float*, int);
__global__ void laplace2(float*, float*, int);
__global__ void init_differential(float*, float*, float*, float*);
__global__ void update_differential(float*, float*, float*, float*, int);
__global__ void update_field(float*, float*, float*, float*, float*);
__global__ void reset(float*, float*, float*, float*, float*);

// MAIN PROGRAM

int main(int argc, char **argv)
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

	int ti = 0, tf = 0;

	// Function Execution

	// set floting-point precision on stdout and stderr
	cout << fixed << setprecision(10);
	cerr << fixed << setprecision(10);

	cout << "Ultrasonic Tomography Running:\n\n";

	ti = clock();
	cout << "ti = " << ti << "\n";

	Ultrasonic_Tomography(group_size, target_epsilon, max_iterations, ti);

	tf = clock();
	cout << "tf = " << tf << "\n"
		 << "tt = " << tf - ti << "\n"
		 << "Total Seconds = " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";

	// End of the program

	return 0;
}

// FUNCTIONS DEFINITION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti)
{
	// Simulation Variables

	float hx = 0.001f;
	float hy = 0.001f;

	int Nx_Ny = NX * NY;
	int Nx_Ny_Nt = NX * NY * NT;
	int Nx_Nt = NX * NT;

	float* x = new float[NX];
	float* y = new float[NY];
	float* fo = new float[Nx_Ny];

	// Kernel Preparation

	/*dim3 Grid_Size(13, 26);
	dim3 Block_Size(16, 8);*/

	/*dim3 Grid_Size(7, 51);
	dim3 Block_Size(32, 4);*/

	/*dim3 Grid_Size(7, 26);
	dim3 Block_Size(32, 8);*/

	dim3 Grid_Size(13, 13);
	dim3 Block_Size(16, 16);

	// Variables of allocation

	float* dev_x;
	int size_x = NX * sizeof(float);

	float* dev_y;
	int size_y = NX * sizeof(float);

	float* dev_fo;
	int size_fo = Nx_Ny * sizeof(float);

	float* dev_v;
	int size_v = Nx_Ny * sizeof(float);

	float* dev_r;
	int size_r = Nx_Ny * sizeof(float);

	float* dev_r2;
	int size_r2 = Nx_Ny * sizeof(float);

	float* dev_s;
	int size_s = Nx_Ny * sizeof(float);

	float* dev_u;
	int size_u = Nx_Ny_Nt * sizeof(float);

	int Ng = NS / group_size;

	float* dev_g1;
	int size_g1 = Nx_Nt * Ng * sizeof(float);

	float* dev_g2;
	int size_g2 = Nx_Nt * Ng * sizeof(float);

	float* dev_g3;
	int size_g3 = Nx_Nt * Ng * sizeof(float);

	float* dev_g4;
	int size_g4 = Nx_Nt * Ng * sizeof(float);

	cudaMalloc((void**) &dev_x, size_x);
	cudaMalloc((void**) &dev_y, size_y);
	cudaMalloc((void**) &dev_fo, size_fo);
	cudaMalloc((void**) &dev_v, size_v);
	cudaMalloc((void**) &dev_r, size_r);
	cudaMalloc((void**) &dev_r2, size_r2);
	cudaMalloc((void**) &dev_s, size_s);
	cudaMalloc((void**) &dev_u, size_u);
	cudaMalloc((void**) &dev_g1, size_g1);
	cudaMalloc((void**) &dev_g2, size_g2);
	cudaMalloc((void**) &dev_g3, size_g3);
	cudaMalloc((void**) &dev_g4, size_g4);

	cudaMemset(dev_u, 0.0, size_u);
	cudaMemset(dev_g1, 0.0, size_g1);
	cudaMemset(dev_g2, 0.0, size_g2);
	cudaMemset(dev_g3, 0.0, size_g3);
	cudaMemset(dev_g4, 0.0, size_g4);

	// Environment Initialization

	for (int i = 0; i < NX; i++)
	{
		x[i] = -0.1f + i * hx;
	}

	for (int j = 0; j < NY; j++)
	{
		y[j] = -0.1f + j * hy;
	}

	cudaMemcpy(dev_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, size_y, cudaMemcpyHostToDevice);

	field_setup<<<Grid_Size, Block_Size>>>(dev_x, dev_y, dev_fo, dev_v, dev_r, dev_r2, dev_s);

	cudaMemcpy(fo, dev_fo, size_fo, cudaMemcpyDeviceToHost);

	// Position of the transducers

	int* jj = new int[NS];
	int* ii = new int[NS];

	for (int p = 0; p < 160; p++)
	{
		jj[p] = 181;
		ii[p] = 21 + (p + 1);
	}

	for (int p = 160; p < 320; p++)
	{
		ii[p] = 181;
		jj[p] = 181 - ((p + 1) - 160);
	}

	for (int p = 320; p < 480; p++)
	{
		jj[p] = 21;
		ii[p] = 181 - ((p + 1) - 320);
	}

	for (int p = 480; p < NS; p++)
	{
		ii[p] = 21;
		jj[p] = 21 + ((p + 1) - 480);
	}

	for (int p = 0; p < NS; p += group_size)
	{
		cudaMemset(dev_u, 0.0, size_u);

		int jp1 = jj[p];
		int jp2 = jj[p + group_size - 1];
		int ip1 = ii[p];
		int ip2 = ii[p + group_size - 1];

		if (jp2 < jp1)
		{
			int jp = jp1;
			jp1 = jp2;
			jp2 = jp;
		}

		if (ip2 < ip1)
		{
			int ip = ip1;
			ip1 = ip2;
			ip2 = ip;
		}

		// Boundary

		for (int k = 1; k < NT - 1; k++)
		{
			propagation<<<Grid_Size, Block_Size>>>(jp1, jp2, ip1, ip2, dev_v, dev_r, dev_r2 , dev_s, dev_u, k);
		}

		// Four corners

		propagation_at_corners<<<1, NT>>>(dev_u);

		initial_signal<<<dim3(NT - 2, 1) , dim3(159, 1)>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, p / group_size);
	}

	// Kaczmarz method
	// propagation

	float* dev_rr1;
	int size_rr1 = Nx_Nt * sizeof(float);

	float* dev_rr2;
	int size_rr2 = Nx_Nt * sizeof(float);

	float* dev_rr3;
	int size_rr3 = Nx_Nt * sizeof(float);

	float* dev_rr4;
	int size_rr4 = Nx_Nt * sizeof(float);

	float* dev_z;
	int size_z = Nx_Ny * (NT + 1) * sizeof(float);

	float* dev_Lu;
	int size_Lu = Nx_Ny_Nt * sizeof(float);

	float* dev_f;
	int size_f = Nx_Ny * sizeof(float);

	float* dev_df;
	int size_df = Nx_Ny * sizeof(float);

	float* dev_alpha;
	int size_alpha = Nx_Ny * sizeof(float);

	float* dev_f_minus_fo;
	int size_f_minus_fo = Nx_Ny * sizeof(float);

	// Allocation

	cudaMalloc((void**) &dev_rr1, size_rr1);
	cudaMalloc((void**) &dev_rr2, size_rr2);
	cudaMalloc((void**) &dev_rr3, size_rr3);
	cudaMalloc((void**) &dev_rr4, size_rr4);
	cudaMalloc((void**) &dev_z, size_z);
	cudaMalloc((void**) &dev_Lu, size_Lu);
	cudaMalloc((void**) &dev_f, size_f);
	cudaMalloc((void**) &dev_df, size_df);
	cudaMalloc((void**) &dev_alpha, size_alpha);
	cudaMalloc((void**) &dev_f_minus_fo, size_f_minus_fo);

	cudaMemset(dev_rr1, 0.0, size_rr1);
	cudaMemset(dev_rr2, 0.0, size_rr2);
	cudaMemset(dev_rr3, 0.0, size_rr3);
	cudaMemset(dev_rr4, 0.0, size_rr4);
	cudaMemset(dev_f, 0.0, size_f);
	cudaMemset(dev_Lu, 0.0, size_Lu);

	float* f = new float[Nx_Ny];
	float* f_minus_fo = new float[Nx_Ny];

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();

	ofstream convergence_file("art_convergence0.txt");
	ofstream time_file("art_time0.txt");

	for (int iter = 0; iter < max_iterations; iter++)
	{
		cout << "\nIter: " << iter << "\n";
		cudaMemset(dev_u, 0.f, size_u);

		for (int p = 0; p < NS; p += group_size)
		{
			int jp1 = jj[p];
			int jp2 = jj[p + group_size - 1];
			int ip1 = ii[p];
			int ip2 = ii[p + group_size - 1];

			if (jp2 < jp1)
			{
				int jp = jp1;
				jp1 = jp2;
				jp2 = jp;
			}

			if (ip2 < ip1)
			{
				int ip = ip1;
				ip1 = ip2;
				ip2 = ip;
			}

			reset<<<Grid_Size, Block_Size>>>(dev_f, dev_v, dev_r, dev_r2, dev_s);

			// Boundary

			for (int k = 1; k < NT - 1; k++)
			{
				propagation<<<Grid_Size, Block_Size>>>(jp1, jp2, ip1, ip2, dev_v, dev_r, dev_r2 , dev_s, dev_u, k);
			}

			// Four corners

			propagation_at_corners<<<1, NT>>>(dev_u);

			difference_signal<<<dim3(NT - 2, 1), dim3(159, 1)>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, dev_rr1, dev_rr2, dev_rr3, dev_rr4, p / group_size);

			cudaMemset(dev_z, 0.0, size_z);

			for (int k = NT - 2; k > 0; k--)
			{
				backpropagation1<<<Grid_Size, Block_Size>>>(dev_z, dev_f, k);
				backpropagation2<<<1, NX>>>(dev_z, dev_rr1, dev_rr2, dev_rr3, dev_rr4, k);
			}

			for (int k = 1; k < NT; k++)
			{
				laplace1<<<Grid_Size, Block_Size>>>(dev_u, dev_Lu, k);
				laplace2<<<1, NX>>>(dev_u, dev_Lu, k);
			}

			init_differential<<<Grid_Size, Block_Size>>>(dev_df, dev_z, dev_Lu, dev_f);

			for (int k = 2; k < NT; k++)
			{
				update_differential<<<Grid_Size, Block_Size>>>(dev_df, dev_z, dev_Lu, dev_f, k);
			}

			update_field<<<Grid_Size, Block_Size>>>(dev_alpha, dev_f, dev_df, dev_f_minus_fo, dev_fo);
		}

		cudaMemcpy(f_minus_fo, dev_f_minus_fo, size_f_minus_fo, cudaMemcpyDeviceToHost);

		curr_epsilon = norm(f_minus_fo, Nx_Ny) / norm(fo, Nx_Ny) * 100.0f;
		float current_t = (float)(clock()-ti) / CLOCKS_PER_SEC;

		convergence_file << curr_epsilon << " ";
		time_file << (current_t)<<" ";

		cout << "epsilon = " << curr_epsilon << "\n";

		// stop if reached target epsilon
		if (curr_epsilon <= target_epsilon) {
			break;
		}

		// stop if epsilon diverges
		if (curr_epsilon > prev_epsilon ||
				std::isnan(curr_epsilon)) {
			break;
		}

		// update prev_epsilon
		prev_epsilon = curr_epsilon;
	}

	cudaMemcpy(f, dev_f, size_f, cudaMemcpyDeviceToHost);

	IO_Files(x, y, fo, f);

	// Free Variables

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_fo);
	cudaFree(dev_v);
	cudaFree(dev_r);
	cudaFree(dev_r2);
	cudaFree(dev_s);
	cudaFree(dev_u);
	cudaFree(dev_g1);
	cudaFree(dev_g2);
	cudaFree(dev_g3);
	cudaFree(dev_g4);
	cudaFree(dev_rr1);
	cudaFree(dev_rr2);
	cudaFree(dev_rr3);
	cudaFree(dev_rr4);
	cudaFree(dev_z);
	cudaFree(dev_Lu);
	cudaFree(dev_f);
	cudaFree(dev_df);
	cudaFree(dev_alpha);
	cudaFree(dev_f_minus_fo);

	delete [] x;
	delete [] y;
	delete [] fo;
	delete [] ii;
	delete [] jj;
	delete [] f;
	delete [] f_minus_fo;

	cudaDeviceReset();
}

__global__ void field_setup(float* x, float* y, float* fo, float* v, float* r, float* r2, float* s)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < NX && j < NY)
	{
		int offset = i + NX * j;
		float value = 0.0f;

		/* if(((sqrtf(powf(x[i] - 0.015f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.015f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f)) */
		/* { */
		/*	value = 0.06f; */
		/* } */
		/* else */
		/* { */
		/*	if(sqrtf(x[i] * x[i] + y[j] * y[j]) <= 0.03f) */
		/*	{ */
		/*		value = 0.02f; */
		/*	} */
		/*	else */
		/*	{ */
		/*		value = 0.0f; */
		/*	} */
		/* } */

		float rc = 0.015f;
		float rp = 0.005f;
		/* float lim = 0.020f; */

		float sc = 0.03f;
		float sp = 0.05f;
		/* float sb = 0.02f; */

		if (powf(x[i], 2) + powf(y[j], 2) <= powf(rc, 2))
		{
			value = sc;
		}

		if (powf(x[i] - rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y[j] - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2))
		{
			value = sp;
		}

		if (powf(x[i] + rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y[j] - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2))
		{
			value = sp;
		}

		if (powf(x[i], 2) + powf(y[j] + rc, 2) <= powf(rp, 2))
		{
			value = sp;
		}

		fo[offset] = value;
		v[offset] = 1500.0f * sqrtf(1.0f + value);
		r[offset] = v[offset] * dt / hx;
		r2[offset] = powf(r[offset], 2.0f);
		s[offset] = 2.0f - 4.0f * r2[offset];

		/*int offset = i + NX * j;
		float value = 0.0f;

		if (((sqrtf(powf(x[i] - 0.05f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.05f, 2.0f) + powf(y[j] + 0.000f, 2.0f))) <= 0.005f))
		{
			value = 0.06f;
		}
		else
		{
			if (sqrtf(x[i] * x[i] + y[j] * y[j]) <= 0.03f)
			{
				value = 0.02f;
			}
			else
			{
				if ((x[i] >= -0.05f) && (x[i] <= 0.05f) && (y[j] >= -0.06f) && (y[j] <= -0.045f))
				{
					value = 0.04f;
				}
				else
				{
					if ((x[i] >= -0.03f) && (x[i] <= 0.00f) && (y[j] <= 0.065f) && (y[j] >= (0.04f - 0.5f * x[i])))
					{
						value = 0.03f;
					}
					else
					{
						if ((x[i] >= 0.00f) && (x[i] <= 0.03f) && (y[j] <= 0.065f) && (y[j] >= (0.04f + 0.5f * x[i])))
						{
							value = 0.03f;
						}
						else
						{
							value = 0.0f;
						}
					}
				}
			}
		}*/

		fo[offset] = value;
		v[offset] = 1500.0f * sqrtf(1.0f + value);
		r[offset] = v[offset] * dt / hx;
		r2[offset] = powf(r[offset], 2.0f);
		s[offset] = 2.0f - 4.0f * r2[offset];
	}
}

__global__ void propagation(int jp1, int jp2, int ip1, int ip2, float* v, float* r, float* r2 , float* s, float* u, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny = NX * NY;
	int Nx_Ny_k = Nx_Ny * k;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		if (((j + 1) >= jp1) && ((j + 1) <= jp2) && ((i + 1) >= ip1) && ((i + 1) <= ip2) && ((k + 1) <= 24))
		{
			u[offset + Nx_Ny * (k + 1)] = powf(v[offset], 2.0f) * powf(dt, 2.0f) * cosf(omegac * (k * dt - tt)) * expf(-powf(k * dt - tt, 2.0f) / (2.0f * powf(tao, 2.0f))) + r2[offset] * (u[(i + 1) + NX * j + Nx_Ny_k] + u[(i - 1) + NX * j + Nx_Ny_k] + u[i + NX * (j - 1) + Nx_Ny_k] + u[i + NX * (j + 1) + Nx_Ny_k]) + s[offset] * u[offset + Nx_Ny_k] - u[offset + Nx_Ny * (k - 1)];
		}
		else
		{
			u[offset + Nx_Ny * (k + 1)] = r2[offset] * (u[(i + 1) + NX * j + Nx_Ny_k] + u[(i - 1) + NX * j + Nx_Ny_k] + u[i + NX * (j - 1) + Nx_Ny_k] + u[i + NX * (j + 1) + Nx_Ny_k]) + s[offset] * u[offset + Nx_Ny_k] - u[offset + Nx_Ny * (k - 1)];
		}

		if ((i == 0) && (j > 0) && (j < (NY - 1)))
		{
			u[offset + Nx_Ny * (k + 1)] = (2.0f - 2.0f * r[offset] - r2[offset]) * u[offset + Nx_Ny_k] + 2.0f * r[offset] * (1.0f + r[offset]) * u[(offset + 1) + Nx_Ny_k] - r2[offset] * u[(offset + 2) + Nx_Ny_k] + (2.0f * r[offset] - 1.0f) * u[offset + Nx_Ny * (k - 1)] - 2.0f * r[offset] * u[(offset + 1) + Nx_Ny * (k - 1)];
		}


		if ((i == NX - 1) && (j > 0) && (j < (NY - 1)))
		{
			u[offset + Nx_Ny * (k + 1)] = (2.0f - 2.0f * r[offset] - r2[offset]) * u[offset + Nx_Ny_k] + 2.0f * r[offset] * (1.0f + r[offset]) * u[(offset - 1) + Nx_Ny_k] - r2[offset] * u[(offset - 2) + Nx_Ny_k] + (2.0f * r[offset] - 1.0f) * u[offset + Nx_Ny * (k - 1)] - 2.0f * r[offset] * u[(offset - 1) + Nx_Ny * (k - 1)];
		}


		if ((j == 0) && (i > 0) && (i < (NX - 1)))
		{
			u[offset + Nx_Ny * (k + 1)] = (2.0f - 2.0f * r[offset] - r2[offset]) * u[offset + Nx_Ny_k] + 2.0f * r[offset] * (1.0f + r[offset]) * u[(i + (j + 1) * NX) + Nx_Ny_k] - r2[offset] * u[(i + (j + 2) * NX) + Nx_Ny_k] + (2.0f * r[offset] - 1.0f) * u[offset + Nx_Ny * (k - 1)] - 2.0f * r[offset] * u[(i + (j + 1) * NX) + Nx_Ny * (k - 1)];
		}


		if ((j == NY - 1) && (i > 0) && (i < (NX - 1)))
		{
			u[offset + Nx_Ny * (k + 1)] = (2.0f - 2.0f * r[offset] - r2[offset]) * u[offset + Nx_Ny_k] + 2.0f * r[offset] * (1.0f + r[offset]) * u[(i + (j - 1) * NX) + Nx_Ny_k] - r2[offset] * u[(i + (j - 2) * NX) + Nx_Ny_k] + (2.0f * r[offset] - 1.0f) * u[offset + Nx_Ny * (k - 1)] - 2.0f * r[offset] * u[(i + (j - 1) * NX) + Nx_Ny * (k - 1)];
		}
	}
}

__global__ void propagation_at_corners(float* u)
{
	int k = threadIdx.x;
	int Nx_Ny = NX * NY;
	int Nx_Ny_k = Nx_Ny * k;

	u[Nx_Ny_k] = 1.0f / 2.0f * (u[NX + k] + u[1 + k]);
	u[(NX - 1) + Nx_Ny_k] = 1.0f / 2.0f * (u[(NX - 2) + Nx_Ny_k] + u[(NX - 1) + NX + Nx_Ny_k]);
	u[(NY - 1) * NX + Nx_Ny_k] = 1.0f / 2.0f * (u[(NY - 2) * NX + Nx_Ny_k] + u[1 +(NY - 1) * NX + Nx_Ny_k]);
	u[(NX - 1) + (NY - 1) * NX + Nx_Ny_k] = 1.0f / 2.0f * (u[(NX - 2) + (NY - 1) * NX + Nx_Ny_k] + u[(NX - 1) + (NY - 2) * NX + Nx_Ny_k]);
}

__global__ void initial_signal(float* u, float* g1, float* g2, float* g3, float* g4, int p)
{
	int i = threadIdx.x + 21;
	int k = blockIdx.x + 2;

	int Nx_Ny_k = NX * NY * k;
	int i_k_Nx_Nx_Nt_p = i + NX * k + NX * NT * p;

	g1[i_k_Nx_Nx_Nt_p] = u[i + NX * 180 + Nx_Ny_k];
	g3[i_k_Nx_Nx_Nt_p] = u[i + NX * 20 + Nx_Ny_k];

	g2[i_k_Nx_Nx_Nt_p] = u[180 + NX * i + Nx_Ny_k];
	g4[i_k_Nx_Nx_Nt_p] = u[20 + NX * i + Nx_Ny_k];
}

__global__ void difference_signal(float* u, float* g1, float* g2, float* g3, float* g4, float* rr1, float* rr2, float* rr3, float* rr4, int p)
{
	int i = threadIdx.x + 21;
	int k = blockIdx.x + 2;

	int Nx_Ny_k = NX * NY * k;
	int i_k_Nx_Nx_Nt_p = i + k * NX + NX * NT * p;
	int i_Nx_k = i + NX * k;

	rr1[i_Nx_k] = g1[i_k_Nx_Nx_Nt_p] - u[i + NX * 180 + Nx_Ny_k];
	rr3[i_Nx_k] = g3[i_k_Nx_Nx_Nt_p] - u[i + NX * 20 + Nx_Ny_k];

	rr2[i_Nx_k] = g2[i_k_Nx_Nx_Nt_p] - u[180 + NX * i + Nx_Ny_k];
	rr4[i_Nx_k] = g4[i_k_Nx_Nx_Nt_p] - u[20 + NX * i + Nx_Ny_k];
}

__global__ void backpropagation1(float* z, float* f, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny = NX * NY;
	int Nx_Ny_k_p_1 = Nx_Ny * (k + 1);

	if((i >= 1) && (i < (NX - 1)) && (j >= 1) && (j < (NY - 1)))
	{
		int offset = i + NX * j;
		int up = i + NX * (j - 1);
		int down = i + NX * (j + 1);
		int left = (i - 1) + NX * j;
		int right = (i + 1) + NX * j;

		z[offset + Nx_Ny * k] = powf(1500.0f, 2.0f) * (dt * dt) * ((1.0f + f[up]) * z[up + Nx_Ny_k_p_1] + (1.0f + f[down]) * z[down + Nx_Ny_k_p_1] + (1.0f + f[left]) * z[left + Nx_Ny_k_p_1] + (1.0f + f[right]) * z[right + Nx_Ny_k_p_1] - 4.0f * (1.0f + f[offset]) * z[offset + Nx_Ny_k_p_1]) / (h * h) + 2.0f * z[offset + Nx_Ny_k_p_1] - z[offset + Nx_Ny * (k + 2)];
	}
}

__global__ void backpropagation2(float* z, float* rr1, float* rr2, float* rr3, float* rr4, int k)
{
	int i = threadIdx.x;
	int Nx_Ny_k = NX * NY * k;
	int i_Nx_k = i + NX * k;

	if((i >= 21) && (i < 180))
	{
		z[i + NX * 180 + Nx_Ny_k] = z[i + NX * 179 + Nx_Ny_k] + rr1[i_Nx_k] * h * 1000.0f;
		z[i + NX * 20 + Nx_Ny_k] = z[i + NX * 21 + Nx_Ny_k] + rr3[i_Nx_k] * h * 1000.0f;

		z[180 + NX * i + Nx_Ny_k] = z[179 + NX * i + Nx_Ny_k] + rr2[i_Nx_k] * h * 1000.0f;
		z[20 + NX * i + Nx_Ny_k] = z[21 + NX * i + Nx_Ny_k] + rr4[i_Nx_k] * h * 1000.0f;
	}

	if((i >= 1) && (i < (NX - 1)))
	{
		z[i + Nx_Ny_k] = z[i + NX + Nx_Ny_k];
		z[i + NX * (NY - 1) + Nx_Ny_k] = z[i + NX * (NY - 2) + Nx_Ny_k];

		z[NX * i + Nx_Ny_k] = z[1 + NX * i + Nx_Ny_k];
		z[(NX - 1) + NX * i + Nx_Ny_k] = z[(NX - 2) + NX * i + Nx_Ny_k];
	}

	if(i == 0)
	{
		z[Nx_Ny_k] = (z[1 + Nx_Ny_k] + z[NX + Nx_Ny_k]) / 2.0f;
		z[(NX - 1) + Nx_Ny_k] = (z[(NX - 2) + Nx_Ny_k] + z[(NX - 1) + NX + Nx_Ny_k]) / 2.0f;
		z[NX * (NY - 1) + Nx_Ny_k] = (z[1 + NX * (NY - 1) + Nx_Ny_k] + z[NX * (NY - 2) + Nx_Ny_k]) / 2.0f;
		z[(NX - 1) + NX * (NY - 1) + Nx_Ny_k] = (z[(NX - 2) + NX * (NY - 1) + Nx_Ny_k] + z[(NX - 1) + NX * (NY - 2) + Nx_Ny_k]) / 2.0f;
	}
}

__global__ void laplace1(float* u, float* Lu, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny = NX * NY;
	int Nx_Ny_k = Nx_Ny * k;

	if((i >= 1) && (i < (NX - 1)) && (j >= 1) && (j < (NY - 1)))
	{
		int offset = i + NX * j;

		Lu[offset + Nx_Ny_k] = (u[i + NX * (j - 1) + Nx_Ny_k] + u[i + NX * (j + 1) + Nx_Ny_k] + u[(i - 1) + NX * j + Nx_Ny_k] + u[(i + 1) + NX * j + Nx_Ny_k] - 4.0f * u[offset + Nx_Ny_k]) / (h * h);
	}
}

__global__ void laplace2(float* u, float* Lu, int k)
{
	int i = threadIdx.x;
	int Nx_Ny_k = NX * NY * k;

	if((i >= 1) && (i < (NX - 1)))
	{
		Lu[i + Nx_Ny_k] = (u[i + Nx_Ny_k] + u[i + NX + Nx_Ny_k] + u[(i - 1) + Nx_Ny_k] + u[(i + 1) + Nx_Ny_k] - 4.0f * u[i + Nx_Ny_k]) / (h * h);
		Lu[i + NX * (NY - 1) + Nx_Ny_k] = (u[i + NX * (NY - 1) + Nx_Ny_k] + u[i + NX * (NY - 2) + Nx_Ny_k] + u[(i - 1) + NX * (NY - 1) + Nx_Ny_k] + u[(i + 1) + NX * (NY - 1) + Nx_Ny_k] - 4.0f * u[i + NX * (NY - 1) + Nx_Ny_k]) / (h * h);

		Lu[NX * i + Nx_Ny_k] = (u[NX * i + Nx_Ny_k] + u[1 + NX * i + Nx_Ny_k] + u[NX * (i - 1) + Nx_Ny_k] + u[NX * (i + 1) + Nx_Ny_k] - 4.0f * u[NX * i + Nx_Ny_k]) / (h * h);
		Lu[(NX - 1) + NX * i + Nx_Ny_k] = (u[(NX - 1) + NX * i + Nx_Ny_k] + u[(NX - 2) + NX * i + Nx_Ny_k] + u[(NX - 1) + NX * (i - 1) + Nx_Ny_k] + u[(NX - 1) + NX * (i + 1) + Nx_Ny_k] - 4.0f * u[(NX - 1) + NX * i + Nx_Ny_k]) / (h * h);
	}

	if(i == 0)
	{
		Lu[Nx_Ny_k] = (Lu[1 + Nx_Ny_k] + Lu[NX + Nx_Ny_k]) / 2.0f;
		Lu[(NX - 1) + Nx_Ny_k] = (Lu[(NX - 2) + Nx_Ny_k] + Lu[(NX - 1) + NX + Nx_Ny_k]) / 2.0f;
		Lu[NX * (NY - 1) + Nx_Ny_k] = (Lu[1 + NX * (NY - 1) + Nx_Ny_k] + Lu[NX * (NY - 2) + Nx_Ny_k]) / 2.0f;
		Lu[(NX - 1) + NX * (NY - 1) + Nx_Ny_k] = (Lu[(NX - 2) + NX * (NY - 1) + Nx_Ny_k] + Lu[(NX - 1) + NX * (NY - 2) + Nx_Ny_k]) / 2.0f;
	}
}

__global__ void init_differential(float* df, float* z, float* Lu, float* f)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny = NX * NY;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		df[offset] = z[offset + Nx_Ny] * Lu[offset + Nx_Ny] / (1.0f + f[offset]);
	}
}

__global__ void update_differential(float* df, float* z, float* Lu, float* f, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny_k = NX * NY * k;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		df[offset] += z[offset + Nx_Ny_k] * Lu[offset + Nx_Ny_k] / (1.0f + f[offset]);
	}
}

__global__ void update_field(float* alpha, float* f, float* df, float* f_minus_fo, float* fo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		if((i >= 21) && (i < 180) && (j >= 21) && (j < 180))
		{
			alpha[offset] = 1.0f;
		}
		else
		{
			alpha[offset] = 0.0f;
		}

		f[offset] += 20000.0f * alpha[offset] * df[offset];

		f_minus_fo[offset] = f[offset] - fo[offset];
	}
}

__global__ void reset(float* f, float* v, float* r, float* r2, float* s)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		v[offset] = 1500.0f * sqrtf(1.0f + f[offset]);
		r[offset] = v[offset] * dt / hx;
		r2[offset] = powf(r[offset], 2.0f);
		s[offset] = 2.0f - 4.0f * r2[offset];
	}
}

void IO_Files(float *x, float *y, float *fo, float *f)
{
	// I/O Files

	ofstream x_file, y_file;
	ofstream fo_file;
	ofstream f_file;

	x_file.open("dev_x.txt");
	y_file.open("dev_y.txt");
	fo_file.open("dev_f0.txt");
	f_file.open("dev_f.txt");

	for (int i = 0; i < NX; i++)
	{
		x_file << x[i];
		x_file << "\n";
	}

	for (int j = 0; j < NX; j++)
	{
		y_file << y[j];
		y_file << "\n";
	}

	for (int j = 0; j < NY; j++)
	{
		for (int i = 0; i < NX; i++)
		{
			fo_file << fo[i + NX * j];
			fo_file << "\t";
		}

		fo_file << "\n";
	}

	for (int j = 0; j < NY; j++)
	{
		for (int i = 0; i < NX; i++)
		{
			f_file << f[i + NX * j];
			f_file << " ";
		}

		f_file << "\n";
	}

	x_file.close();
	y_file.close();
	fo_file.close();
	f_file.close();
}

float norm(float* A, int lenght)
{
	float sum = 0;

	for (int i = 0; i < lenght; i++)
	{
		sum += A[i] * A[i];
	}

	return sqrtf(sum);
}
