// HEADERS

#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;

// DEFINITIONS

#define NX 192				//was 201
#define PX 192				//was 224
#define NY 192				//was 201
#define PY 192				//was 224
#define NT 401

#define NS 640 							//number of sensors

#define BLOCK_X 16
#define BLOCK_Y 16

__constant__ float hx = 0.001f;
__constant__ float hy = 0.001f; // pixel size
__constant__ float h = 0.001f;

/* __constant__ float T = 1.3333e-04f; // 0.2f / 1500.f; */
__constant__ float dt = 3.3333e-07f; // T / 400.f;
/* __constant__ float fre = 125000.f; */
__constant__ float omegac = 7.8540e+05f; // 2.f * pi * fre; // wavelength
__constant__ float tao = 4.0000e-06f; // pi / omegac;
__constant__ float tt = 8.1573e-06f; // sqrtf(6.f * logf(2.f)) * tao; // time delay

// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti);
void Position_Transducers (int *&, int *&, int);
void IO_Files(float*, float*, float*, float*);
float norm(float*, int, int);

__global__ void field_setup(const float*, const float*, float*);
__global__ void propagation(int, int, int, int, const float*, float*, int);
__global__ void propagation_at_corners(float*);
__global__ void initial_signal(const float*, float*, float*, float*, float*, int);
__global__ void difference_signal(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, int);
__global__ void backpropagation1(float*, const float*, int);
__global__ void backpropagation2(float*, const float*, const float*, const float*, const float*, int);
__global__ void laplace(const float*, float*);
__global__ void laplace_corners(const float*, float*);
__global__ void update_differential(float*, const float*, const float*, const float*);
__global__ void update_field(float*, const float*, float*, const float*);

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

inline
int grid_size(int n, int threads)
{
	return ceil(float(n) / threads);
}

// FUNCTIONS DEFINITION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti)
{
	// Simulation Variables

	float hx = 0.001f;
	float hy = 0.001f;

	int i = 0, j = 0, k = 0;

	float *x = new float[PX];
	float *y = new float[PY];
	float *fo = new float[PX * PY];

	// Kernel Preparation

	/*dim3 Grid_Size(13, 26);
	dim3 Block_Size(16, 8);*/

	/*dim3 Grid_Size(7, 51);
	dim3 Block_Size(32, 4);*/

	/*dim3 Grid_Size(7, 26);
	dim3 Block_Size(32, 8);*/

	dim3 Block_Size(BLOCK_X, BLOCK_Y);
	dim3 Grid_Size(grid_size(PX, BLOCK_X), grid_size(PY, BLOCK_Y));

	// Variables of allocation

	float *dev_x;
	int size_x = PX * sizeof(float);

	float *dev_y;
	int size_y = PX * sizeof(float);

	float *dev_fo;
	int size_fo = PX * PY * sizeof(float);

	float *dev_u;
	int size_u = PX * PY * NT * sizeof(float);


	float *dev_g1;
	float *dev_g2;
	float *dev_g3;
	float *dev_g4;
	int size_g = PX * NT * (NS / group_size) * sizeof(float);

	cudaMalloc((void**) &dev_x, size_x);
	cudaMalloc((void**) &dev_y, size_y);
	cudaMalloc((void**) &dev_fo, size_fo);
	cudaMalloc((void**) &dev_u, size_u);
	cudaMalloc((void**) &dev_g1, size_g);
	cudaMalloc((void**) &dev_g2, size_g);
	cudaMalloc((void**) &dev_g3, size_g);
	cudaMalloc((void**) &dev_g4, size_g);

	cudaMemset(dev_u, 0.f, size_u);
	cudaMemset(dev_g1, 0.f, size_g);
	cudaMemset(dev_g2, 0.f, size_g);
	cudaMemset(dev_g3, 0.f, size_g);
	cudaMemset(dev_g4, 0.f, size_g);

	// Environment Initialization

	for(i = 0; i < NX; i++)
	{
		x[i] = -0.1f + i * hx;
	}

	for(j = 0; j < NY; j++)
	{
		y[j] = -0.1f + j * hy;
	}

	cudaMemcpy(dev_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, size_y, cudaMemcpyHostToDevice);

	field_setup<<<Grid_Size, Block_Size>>>(dev_x, dev_y, dev_fo);

	cudaMemcpy(fo, dev_fo, size_fo, cudaMemcpyDeviceToHost);

	// Position of the transducers

	int *ii, *jj;
	Position_Transducers (ii, jj, NS);

	dim3 threads_propagation(16, 16);
	dim3 grid_propagation(grid_size(PX, threads_propagation.x),
							grid_size(PY, threads_propagation.y));

	int p;
	for(p = 0; p < NS; p += group_size)
	{
		cudaMemset(dev_u, 0.f, size_u);

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

		for(k = 1; k < NT - 1; k++)
		{
			propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, dev_fo, dev_u, k);
		}

		// Four corners

		propagation_at_corners<<<1, NT>>>(dev_u);

		initial_signal<<<NT - 2, 159>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, p / group_size);
	}

	// Kaczmarz method
	// propagation

	float *dev_rr1;
	int size_rr1 = PX * NT * sizeof(float);

	float *dev_rr2;
	int size_rr2 = PX * NT * sizeof(float);

	float *dev_rr3;
	int size_rr3 = PX * NT * sizeof(float);

	float *dev_rr4;
	int size_rr4 = PX * NT * sizeof(float);

	float *dev_z;
	int size_z = PX * PY * (NT + 1) * sizeof(float);

	float *dev_Lu;
	int size_Lu = PX * PY * NT * sizeof(float);

	float *dev_f;
	int size_f = PX * PY * sizeof(float);

	float *dev_df;
	int size_df = PX * PY * sizeof(float);

	float *dev_f_minus_fo;
	int size_f_minus_fo = PX * PY * sizeof(float);

	// Allocation

	cudaMalloc((void**) &dev_rr1, size_rr1);
	cudaMalloc((void**) &dev_rr2, size_rr2);
	cudaMalloc((void**) &dev_rr3, size_rr3);
	cudaMalloc((void**) &dev_rr4, size_rr4);
	cudaMalloc((void**) &dev_z, size_z);
	cudaMalloc((void**) &dev_Lu, size_Lu);
	cudaMalloc((void**) &dev_f, size_f);
	cudaMalloc((void**) &dev_df, size_df);
	cudaMalloc((void**) &dev_f_minus_fo, size_f_minus_fo);

	cudaMemset(dev_rr1, 0.f, size_rr1);
	cudaMemset(dev_rr2, 0.f, size_rr2);
	cudaMemset(dev_rr3, 0.f, size_rr3);
	cudaMemset(dev_rr4, 0.f, size_rr4);
	cudaMemset(dev_f, 0.f, size_f);
	cudaMemset(dev_Lu, 0.f, size_Lu);

	float *f = new float[PX * PY];
	float *f_minus_fo = new float[PX * PY];

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'art_convergence.txt'...\n"
		 << "writing time to 'art_time.txt'...\n";

	ofstream convergence_file("art_convergence.txt");
	ofstream time_file("art_time.txt");

	dim3 threads_backpropagation1(16, 16);
	dim3 grid_backpropagation1(grid_size(PX, threads_backpropagation1.x),
								grid_size(PY, threads_backpropagation1.y));

	dim3 threads_laplace(16, 16, 1);
	dim3 grid_laplace(grid_size(PX, threads_laplace.x),
						grid_size(PY, threads_laplace.y),
						grid_size(NT, threads_laplace.z));

	dim3 threads_differential(16, 16, 1);
	dim3 grid_differential(grid_size(PX, threads_differential.x),
							grid_size(PY, threads_differential.y),
							grid_size(NT, threads_differential.z));

	for(int iter = 0; iter < max_iterations; iter++)
	{
		cout << "\nIter: " << iter << "\n";
		cudaMemset(dev_u, 0.f, size_u);

		for(p = 0; p < NS; p += group_size)
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

			// Boundary

			for(k = 1; k < NT - 1; k++)
			{
				propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, dev_f, dev_u, k);
			}

			// Four corners

			propagation_at_corners<<<1, NT>>>(dev_u);
			difference_signal<<<NT - 2, 159>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, dev_rr1, dev_rr2, dev_rr3, dev_rr4, p / group_size);

			cudaMemset(dev_z, 0.f, size_z);

			for(k = NT - 2; k > 0; k--)
			{
				backpropagation1<<<grid_backpropagation1, threads_backpropagation1>>>(dev_z, dev_f, k);
				backpropagation2<<<1, NX>>>(dev_z, dev_rr1, dev_rr2, dev_rr3, dev_rr4, k);
			}

			laplace<<<grid_laplace, threads_laplace>>>(dev_u, dev_Lu);
			laplace_corners<<<grid_size(NT, 32), 32>>>(dev_u, dev_Lu);

			cudaMemset(dev_df, 0.f, size_df);
			update_differential<<<grid_differential, threads_differential>>>(dev_df, dev_z, dev_Lu, dev_f);

			update_field<<<Grid_Size, Block_Size>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo);
		}

		cudaMemcpy(f_minus_fo, dev_f_minus_fo, size_f_minus_fo, cudaMemcpyDeviceToHost);

		curr_epsilon = norm(f_minus_fo, NX, NY) / norm(fo, NX, NY) * 100.f;
		float current_t = (float)(clock()-ti) / CLOCKS_PER_SEC;

		convergence_file << curr_epsilon << " ";
		time_file << current_t << " ";

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
void Position_Transducers(int *&ii, int *&jj, int num)
{
//returns the (x,y) coordinates of the number of total transducers
	int p = 0;
	ii = (int*)malloc(num * sizeof(int));
	jj = (int*)malloc(num * sizeof(int));


	for(p = 0; p < NS/4; p++)
	{
		ii[p] = 21 + (p + 1);
		jj[p] = 181;
	}

	for(p = 160; p < 320; p++)
	{
		ii[p] = 181;
		jj[p] = 181 - ((p + 1) - 160);
	}

	for(p = 320; p < 480; p++)
	{
		ii[p] = 181 - ((p + 1) - 320);
		jj[p] = 21;
	}

	for(p = 480; p < num; p++)
	{
		ii[p] = 21;
		jj[p] = 21 + ((p + 1) - 480);
	}
}
__global__ void field_setup(const float *x, const float *y, float *fo)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + PX * j;
		float value = 0.f;

		/* if(((sqrtf(powf(x[i] - 0.015f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.015f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f)) */
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
		/*		value = 0.f; */
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


		/*int offset = i + NX * j;
		float value = 0.f;

		if (((sqrtf(powf(x[i] - 0.05f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.05f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f))
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
							value = 0.f;
						}
					}
				}
			}
		}

		fo[offset] = value;
		v[offset] = 1500.f * sqrtf(1.f + value);
		r[offset] = v[offset] * dt / hx;
		r2[offset] = powf(r[offset], 2.f);
		s[offset] = 2.f - 4.f * r2[offset];
		*/
	}
}

__global__ void propagation(int jp1, int jp2, int ip1, int ip2, const float *f, float *u, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i < NX && j < NY) {
		float v = 1500.f * sqrtf(1.f + f[i + PX * j]);
		float r = v * dt / hx;
		float s = 2.f - 4.f * r * r;

		float val; // wil hold new u at (i, j, k + 1)

		// not at boundary
		if (i != 0 && i != NX - 1 && j != 0 && j != NY - 1) {

			// update val
			val =
				r * r *
				(u[(i + 1) + PX * j + PX * PY * k] +
				 u[(i - 1) + PX * j + PX * PY * k] +
				 u[i + PX * (j - 1) + PX * PY * k] +
				 u[i + PX * (j + 1) + PX * PY * k]) +
				s * u[i + PX * j + PX * PY * k] -
				u[i + PX * j + PX * PY * (k - 1)];

			// at sensor, k <= 24
			if ((j + 1) >= jp1 && (j + 1) <= jp2 && (i + 1) >= ip1 && (i + 1) <= ip2 && (k + 1) <= 24) {
				float t = k * dt - tt;

				// add wave value
				val +=
					v * v * dt * dt *
					cosf(omegac * t) *
					expf(-(t * t) / (2.f * tao * tao));
			}
		}

		// at boundary
		else {

			// index variables for different boundary cases
			// TODO: need better names
			int i_A, i_B, j_A, j_B;

			// left boundary
			if (i == 0)
			{
				i_A = i + 1;
				j_A = j;
				i_B = i + 2;
				j_B = j;
			}

			// right boundary
			else if (i == NX - 1)
			{
				i_A = i - 1;
				j_A = j;
				i_B = i - 2;
				j_B = j;
			}

			// top boundary
			else if (j == 0)
			{
				i_A = i;
				j_A = j + 1;
				i_B = i;
				j_B = j + 2;
			}

			// bottom boundary
			else
			{
				i_A = i;
				j_A = j - 1;
				i_B = i;
				j_B = j - 2;
			}

			val =
				(2.f - 2.f * r - r * r) * u[i + PX * j + PX * PY * k] +
				2.f * r * (1.f + r) * u[i_A + PX * j_A + PX * PY * k] -
				r * r * u[i_B + PX * j_B + PX * PY * k] +
				(2.f * r - 1.f) * u[i + PX * j + PX * PY * (k - 1)] -
				2.f * r * u[i_A + PX * j_A + PX * PY * (k - 1)];
		}

		// update u at (i, j, k + 1)
		u[i + PX * j + PX * PY * (k + 1)] = val;
	}
}

__global__ void propagation_at_corners(float *u)
{
	int k = threadIdx.x;

	u[PX * PY * k] =
		1.f / 2.f * (u[PX + k] + u[1 + k]);

	u[(NX - 1) + PX * PY * k] =
		1.f / 2.f * (u[(NX - 2) + PX * PY * k] + u[(NX - 1) + PX + PX * PY * k]);

	u[PX * (NY - 1) + PX * PY * k] =
		1.f / 2.f * (u[PX * (NY - 2) + PX * PY * k] + u[1 + PX * (NY - 1) + PX * PY * k]);

	u[(NX - 1) + PX * (NY - 1) + PX * PY * k] =
		1.f / 2.f * (u[(NX - 2) + PX * (NY - 1) + PX * PY * k] + u[(NX - 1) + PX * (NY - 2) + PX * PY * k]);
}

__global__ void initial_signal(const float *u, float *g1, float *g2, float *g3, float *g4, int p)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	g1[(i + 21) + PX * (k + 2) + PX * NT * p] =
		u[(i + 21) + PX * 180 + PX * PY * (k + 2)];

	g3[(i + 21) + PX * (k + 2) + PX * NT * p] =
		u[(i + 21) + PX * 20 + PX * PY * (k + 2)];

	g2[(i + 21) + PX * (k + 2) + PX * NT * p] =
		u[180 + PX * (i + 21) + PX * PY * (k + 2)];

	g4[(i + 21) + PX * (k + 2) + PX * NT * p] =
		u[20 + PX * (i + 21) + PX * PY * (k + 2)];
}

__global__ void difference_signal(const float *u, const float *g1, const float *g2, const float *g3, const float *g4, float *rr1, float *rr2, float *rr3, float *rr4, int p)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	rr1[(i + 21) + PX * (k + 2)] =
		g1[(i + 21) + PX * (k + 2) + PX * NT * p] -
		u[(i + 21) + PX * 180 + PX * PY * (k + 2)];

	rr3[(i + 21) + PX * (k + 2)] =
		g3[(i + 21) + PX * (k + 2) + PX * NT * p] -
		u[(i + 21) + PX * 20 + PX * PY * (k + 2)];

	rr2[(i + 21) + PX * (k + 2)] =
		g2[(i + 21) + PX * (k + 2) + PX * NT * p] -
		u[180 + PX * (i + 21) + PX * PY * (k + 2)];

	rr4[(i + 21) + PX * (k + 2)] =
		g4[(i + 21) + PX * (k + 2) + PX * NT * p] -
		u[20 + PX * (i + 21) + PX * PY * (k + 2)];
}

__global__ void backpropagation1(float *z, const float *f, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i >= 1) && (i < (NX - 1)) && (j >= 1) && (j < (NY - 1)))
	{
		z[i + PX * j + PX * PY * k] =
			1500.f * 1500.f * (dt * dt) *
			((1.f + f[i + PX * (j - 1)]) * z[i + PX * (j - 1) + PX * PY * (k + 1)] +
			 (1.f + f[i + PX * (j + 1)]) * z[i + PX * (j + 1) + PX * PY * (k + 1)] +
			 (1.f + f[(i - 1) + PX * j]) * z[(i - 1) + PX * j + PX * PY * (k + 1)] +
			 (1.f + f[(i + 1) + PX * j]) * z[(i + 1) + PX * j + PX * PY * (k + 1)] -
			 4.f * (1.f + f[i + PX * j]) *
			 z[i + PX * j + PX * PY * (k + 1)]) / (h * h) +
			2.f * z[i + PX * j + PX * PY * (k + 1)] -
			z[i + PX * j + PX * PY * (k + 2)];
	}
}

__global__ void backpropagation2(float *z, const float *rr1, const float *rr2, const float *rr3, const float *rr4, int k)
{
	int i = threadIdx.x;

	if((i >= 21) && (i < 180))
	{
		z[i + PX * 180 + PX * PY * k] =
		// z[k][180][i]
			z[i + PX * 179 + PX * PY * k] +
			// z[k][179][i]
			rr1[i + PX * k] * h * 1000.f;
			// rr1[k][i]

		z[i + PX * 20 + PX * PY * k] =
		// z[k][20][i]
			z[i + PX * 21 + PX * PY * k] +
			// z[k][21][i]
			rr3[i + PX * k] * h * 1000.f;
			// z[k][i]

		z[180 + PX * i + PX * PY * k] =
		// z[k][i][180]
			z[179 + PX * i + PX * PY * k] +
			// z[k][i][179]
			rr2[i + PX * k] * h * 1000.f;
			// rr2[k][i]

		z[20 + PX * i + PX * PY * k] =
		// z[k][i][20]
			z[21 + PX * i + PX * PY * k] +
			// z[k][i][21]
			rr4[i + PX * k] * h * 1000.f;
			// rr4[k][i]
	}

	if((i >= 1) && (i < (NX - 1)))
	{
		z[i + PX * PY * k] =
		// z[k][0][i]
			z[i + PX + PX * PY * k];
			// z[k][1][i]

		z[i + PX * (NY - 1) + PX * PY * k] =
		// z[k][NY - 1][i]
			z[i + PX * (NY - 2) + PX * PY * k];
			// z[k][NY - 2][i]

		z[PX * i + PX * PY * k] =
		// z[k][i][0]
			z[1 + PX * i + PX * PY * k];
			// z[k][i][1]

		z[(NX - 1) + PX * i + PX * PY * k] =
		// z[k][i][NX - 1]
			z[(NX - 2) + PX * i + PX * PY * k];
			// z[k][i][NX - 2]
	}

	else if(i == 0)
	{
		z[PX * PY * k] =
			(z[1 + PX * PY * k] +
			 z[PX + PX * PY * k]) / 2.f;
			// z[k][1][0]

		z[(NX - 1) + PX * PY * k] =
			(z[(NX - 2) + PX * PY * k] +
			 z[(NX - 1) + PX + PX * PY * k]) / 2.f;
			// z[k][1][NX - 1]

		z[PX * (NY - 1) + PX * PY * k] =
			(z[1 + PX * (NY - 1) + PX * PY * k] +
			// z[k][NY - 1][1]
			 z[PX * (NY - 2) + PX * PY * k]) / 2.f;
			// z[k][NY - 2][0]

		z[(NX - 1) + PX * (NY - 1) + PX * PY * k] =
			(z[(NX - 2) + PX * (NY - 1) + PX * PY * k] +
			// z[k][NY - 1][NX - 2]
			 z[(NX - 1) + PX * (NY - 2) + PX * PY * k]) / 2.f;
			// z[k][NY - 2][NX - 1]
	}
}

__global__ void laplace(const float *u, float *Lu)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i < NX && j < NY && (k + 1) < NT) {

		int j_prev = (j > 0) ? j - 1 : j;
		int j_next = (j < NY - 1) ? j + 1 : j;

		int i_prev = (i > 0) ? i - 1 : i;
		int i_next = (i < NX - 1) ? i + 1 : i;

		Lu[i + PX * j + PX * PY * (k + 1)] =
			(u[i + PX * j_prev + PX * PY * (k + 1)] +
			 u[i + PX * j_next + PX * PY * (k + 1)] +
			 u[i_prev + PX * j + PX * PY * (k + 1)] +
			 u[i_next + PX * j + PX * PY * (k + 1)] -
			 4.f * u[i + PX * j + PX * PY * (k + 1)]) / (h * h);
	}
}

__global__ void laplace_corners(const float *u, float *Lu)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if ((k + 1) < NT) {
		Lu[PX * PY * (k + 1)] =
			(Lu[1 + PX * PY * (k + 1)] +
			 Lu[PX + PX * PY * (k + 1)]) / 2.f;

		Lu[(NX - 1) + PX * PY * (k + 1)] =
			(Lu[(NX - 2) + PX * PY * (k + 1)] +
			 Lu[(NX - 1) + PX + PX * PY * (k + 1)]) / 2.f;

		Lu[PX * (NY - 1) + PX * PY * (k + 1)] =
			(Lu[1 + PX * (NY - 1) + PX * PY * (k + 1)] +
			 Lu[PX * (NY - 2) + PX * PY * (k + 1)]) / 2.f;

		Lu[(NX - 1) + PX * (NY - 1) + PX * PY * (k + 1)] =
			(Lu[(NX - 2) + PX * (NY - 1) + PX * PY * (k + 1)] +
			 Lu[(NX - 1) + PX * (NY - 2) + PX * PY * (k + 1)]) / 2.f;
	}
}

__global__ void update_differential(float *df, const float *z, const float *Lu, const float *f)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < NX && j < NY && (k + 1) < NT) {

		atomicAdd(&df[i + PX * j],
					z[i + PX * j + PX * PY * (k + 1)] *
					Lu[i + PX * j + PX * PY * (k + 1)] /
					(1.f + f[i + PX * j]));
	}
}

__global__ void update_field(float *f, const float *df, float *f_minus_fo, const float *fo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + PX * j;

		bool in_sensor_field = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

		float alpha = in_sensor_field ? 1.f : 0.f;

		f[offset] += 20000.f * alpha * df[offset];
		f_minus_fo[offset] = f[offset] - fo[offset];
	}
}

void IO_Files(float *x, float *y, float *fo, float *f)
{
	int i = 0, j = 0;

	// I/O Files

	ofstream x_file, y_file;
	ofstream fo_file;
	ofstream f_file;

	cerr << "writing x to 'dev_x.txt'...\n"
		 << "writing y to 'dev_y.txt'...\n"
		 << "writing f0 to 'dev_f0.txt'...\n"
		 << "writing f to 'dev_f.txt'...\n\n";

	x_file.open("dev_x.txt");
	y_file.open("dev_y.txt");
	fo_file.open("dev_f0.txt");
	f_file.open("dev_f.txt");

	for(i = 0; i < NX; i++)
	{
		x_file << x[i];
		x_file << "\n";
	}

	for(j = 0; j < NX; j++)
	{
		y_file << y[j];
		y_file << "\n";
	}

	for(j = 0; j < NY; j++)
	{
		for(i = 0; i < NX; i++)
		{
			fo_file << fo[i + PX * j];
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(j = 0; j < NY; j++)
	{
		for(i = 0; i < NX; i++)
		{
			f_file << f[i + PX * j];
			f_file << " ";
		}

		f_file << "\n";
	}

	x_file.close();
	y_file.close();
	fo_file.close();
	f_file.close();
}

float norm(float *A, int nx, int ny)
{
	float sum = 0;

	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i)
			sum += A[i + PX * j] * A[i + PX * j];

	return sqrtf(sum);
}
