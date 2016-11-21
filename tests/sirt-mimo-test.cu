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
#define NY 192				//was 201
#define NT 401

#define NS 640 				//number of sensors

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
__global__ void propagation(int, int, int, int, const float*, float*, int, int);
__global__ void propagation_at_corners(float*);
__global__ void propagation_at_corners(float*, int);
__global__ void initial_signal(const float*, float*, float*, float*, float*, int);
__global__ void difference_signal(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, int);
__global__ void backpropagation1(float*, const float*, int, int);
__global__ void backpropagation2(float*, const float*, const float*, const float*, const float*, int, int);
__global__ void laplace(const float*, float*, int);
__global__ void laplace_corners(const float*, float*, int);
__global__ void update_differential(float*, const float*, const float*, const float*, int);
__global__ void print_differential(const float *df);
__global__ void update_field(float*, const float*, float*, const float*);

template <typename T>
__host__ __device__
T& get(T* ptr, int i, int j = 0, int k = 0, int nx = NX, int ny = NY)
{
	return ptr[i + nx * j + nx * ny * k];
}

template <typename T>
__host__ __device__
T& getn(T* ptr, int i, int j = 0, int k = 0, int l = 0, int nx = NX, int ny = NY, int nz = NT)
{
	return ptr[i + nx * j + nx * ny * k + nx * ny * nz * l];
}

__host__ __device__
inline int grid_size(int n, int threads)
{
	return ceil((float)(n) / threads);
}

__global__ void pre_sirt(
		const int group_size, const int num_groups,
		const int *ii, const int *jj,
		const float *fo, float *u,
		float *g_bottom, float *g_top,
		float *g_left, float *g_right
		)
{
	// unique id for each group
	int g = threadIdx.x + blockIdx.x * blockDim.x;

	if (g >= num_groups)
		return;

	int p = g * group_size;

	int jp1 = jj[p];
	int jp2 = jj[p + group_size - 1];
	int ip1 = ii[p];
	int ip2 = ii[p + group_size - 1];

	if (jp2 < jp1) {
		int jp = jp1;
		jp1 = jp2;
		jp2 = jp;
	}

	if (ip2 < ip1) {
		int ip = ip1;
		ip1 = ip2;
		ip2 = ip;
	}

	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
		grid_size(NX, threads_propagation.x),
		grid_size(NY, threads_propagation.y));

	for(int k = 1; k < NT - 1; k++)
		propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, fo, u, k, g);

	propagation_at_corners<<<NT, 1>>>(u, g);

	initial_signal<<<NT - 2, 159>>>(u, g_bottom, g_right, g_top, g_left, g);
}

__global__ void sirt(
		const int group_size, const int num_groups,
		const int *ii, const int *jj,
		float *f, const float *fo, float *u,
		const float *g_bottom, const float *g_top,
		const float *g_left, const float *g_right,
		float *rr_bottom, float *rr_top,
		float *rr_left, float *rr_right,
		float *z, float *Lu, float *df, float *f_minus_fo
		)
{
	// unique id for each group
	int g = threadIdx.x + blockIdx.x * blockDim.x;

	if (g >= num_groups)
		return;

	int p = g * group_size;

	int jp1 = jj[p];
	int jp2 = jj[p + group_size - 1];
	int ip1 = ii[p];
	int ip2 = ii[p + group_size - 1];

	if (jp2 < jp1) {
		int jp = jp1;
		jp1 = jp2;
		jp2 = jp;
	}

	if (ip2 < ip1) {
		int ip = ip1;
		ip1 = ip2;
		ip2 = ip;
	}

	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
		grid_size(NX, threads_propagation.x),
		grid_size(NY, threads_propagation.y));

	dim3 threads_backpropagation1(NX, 1, 1);
	dim3 grid_backpropagation1(
		grid_size(NX, threads_backpropagation1.x),
		grid_size(NY, threads_backpropagation1.y));

	dim3 threads_laplace(96, 2, 1);
	dim3 grid_laplace(
		grid_size(NX, threads_laplace.x),
		grid_size(NY, threads_laplace.y),
		grid_size(NT, threads_laplace.z));

	dim3 threads_differential(96, 2, 1);
	dim3 grid_differential(
		grid_size(NX, threads_differential.x),
		grid_size(NY, threads_differential.y),
		grid_size(NT, threads_differential.z));

	for (int k = 1; k < NT - 1; ++k)
		propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, f, u, k, g);

	propagation_at_corners<<<NT, 1>>>(u, g);
	difference_signal<<<NT - 2, 159>>>(u, g_bottom, g_right, g_top, g_left, rr_bottom, rr_right, rr_top, rr_left, g);

	for (int k = NT - 2; k > 0; --k) {
		backpropagation1<<<grid_backpropagation1, threads_backpropagation1>>>(z, f, k, g);
		backpropagation2<<<NX, 1>>>(z, rr_bottom, rr_right, rr_top, rr_left, k, g);
	}

	laplace<<<grid_laplace, threads_laplace>>>(u, Lu, g);
	laplace_corners<<<NT, 1>>>(u, Lu, g);

	update_differential<<<grid_differential, threads_differential>>>(df, z, Lu, f, g);
}


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

	// set floating-point precision on stdout and stderr
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

	// number of sensor groups that will be launched in parallel
	int num_groups = NS / group_size;

	// Simulation Variables

	float hx = 0.001f;
	float hy = 0.001f;

	int i = 0, j = 0;

	float *x = new float[NX];
	float *y = new float[NY];
	float *fo = new float[NX * NY];


	dim3 Block_Size(BLOCK_X, BLOCK_Y);
	dim3 Grid_Size(grid_size(NX, BLOCK_X), grid_size(NY, BLOCK_Y));

	// Variables of allocation

	float *dev_x;
	int size_x = NX * sizeof(float);

	float *dev_y;
	int size_y = NY * sizeof(float);

	float *dev_fo;
	int size_fo = NX * NY * sizeof(float);

	float *dev_u;
	int size_u = NX * NY * NT * num_groups * sizeof(float);


	float *dev_g_bottom;
	float *dev_g_right;
	float *dev_g_top;
	float *dev_g_left;
	int size_g = NX * NT * num_groups * sizeof(float);


	cudaMalloc((void**) &dev_x, size_x);
	cudaMalloc((void**) &dev_y, size_y);
	cudaMalloc((void**) &dev_fo, size_fo);
	cudaMalloc((void**) &dev_u, size_u);
	cudaMalloc((void**) &dev_g_bottom, size_g);
	cudaMalloc((void**) &dev_g_right, size_g);
	cudaMalloc((void**) &dev_g_top, size_g);
	cudaMalloc((void**) &dev_g_left, size_g);

	cudaMemset(dev_u, 0.f, size_u);
	cudaMemset(dev_g_bottom, 0.f, size_g);
	cudaMemset(dev_g_right, 0.f, size_g);
	cudaMemset(dev_g_top, 0.f, size_g);
	cudaMemset(dev_g_left, 0.f, size_g);

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
	Position_Transducers(ii, jj, NS);

	int *dev_ii;
	int *dev_jj;

	cudaMalloc((void**) &dev_ii, NS * sizeof(int));
	cudaMalloc((void**) &dev_jj, NS * sizeof(int));
	cudaMemcpy(dev_ii, ii, NS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jj, jj, NS * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
		grid_size(NX, threads_propagation.x),
		grid_size(NY, threads_propagation.y));

	cudaMemset(dev_u, 0.f, size_u);

	pre_sirt<<<1, num_groups>>>(
		group_size, num_groups,
		dev_ii, dev_jj,
		dev_fo, dev_u,
		dev_g_bottom, dev_g_top,
		dev_g_left, dev_g_right);

	// Kaczmarz method
	// propagation

	float *dev_rr_bottom;
	float *dev_rr_right;
	float *dev_rr_top;
	float *dev_rr_left;
	int size_rr = NX * NT * num_groups * sizeof(float);

	float *dev_z;
	int size_z = NX * NY * (NT + 1) * num_groups * sizeof(float);

	float *dev_Lu;
	int size_Lu = NX * NY * NT * num_groups * sizeof(float);

	float *dev_f;
	int size_f = NX * NY * sizeof(float);

	float *dev_df;
	int size_df = NX * NY * sizeof(float);

	float *dev_f_minus_fo;
	int size_f_minus_fo = NX * NY * sizeof(float);

	// Allocation

	cudaMalloc((void**) &dev_rr_bottom, size_rr);
	cudaMalloc((void**) &dev_rr_right, size_rr);
	cudaMalloc((void**) &dev_rr_top, size_rr);
	cudaMalloc((void**) &dev_rr_left, size_rr);
	cudaMalloc((void**) &dev_z, size_z);
	cudaMalloc((void**) &dev_Lu, size_Lu);
	cudaMalloc((void**) &dev_f, size_f);
	cudaMalloc((void**) &dev_df, size_df);
	cudaMalloc((void**) &dev_f_minus_fo, size_f_minus_fo);

	cudaMemset(dev_rr_bottom, 0.f, size_rr);
	cudaMemset(dev_rr_right, 0.f, size_rr);
	cudaMemset(dev_rr_top, 0.f, size_rr);
	cudaMemset(dev_rr_left, 0.f, size_rr);
	cudaMemset(dev_f, 0.f, size_f);
	cudaMemset(dev_Lu, 0.f, size_Lu);

	float *f = new float[NX * NY];
	float *f_minus_fo = new float[NX * NY];

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'sirt_convergence.txt'...\n"
		 << "writing time to 'sirt_time.txt'...\n";

	ofstream convergence_file("sirt_convergence.txt");
	ofstream time_file("sirt_time.txt");

	for(int iter = 0; iter < max_iterations; iter++)
	{
		cout << "\nIter: " << iter << "\n";
		cudaMemset(dev_u, 0.f, size_u);
		cudaMemset(dev_z, 0.f, size_z);
		cudaMemset(dev_df, 0.f, size_df);

		sirt<<<1, num_groups>>>
			(group_size, num_groups,
			 dev_ii, dev_jj,
			 dev_f, dev_fo, dev_u,
			 dev_g_bottom, dev_g_top, dev_g_left, dev_g_right,
			 dev_rr_bottom, dev_rr_top, dev_rr_left, dev_rr_right,
			 dev_z, dev_Lu, dev_df, dev_f_minus_fo);

		/* print_differential<<<1, 1>>>(dev_df); */

		update_field<<<Grid_Size, Block_Size>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo);

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

	cout << endl;

	cudaError_t error = cudaGetLastError();

	if (error != cudaSuccess) {
		cerr << cudaGetErrorString(error) << endl;
	}

	cudaMemcpy(f, dev_f, size_f, cudaMemcpyDeviceToHost);

	IO_Files(x, y, fo, f);

	// Free Variables

	size_t free, total;
	cudaMemGetInfo(&free, &total);

	cerr << fixed << setprecision(4);

	cerr << "used mem:  " << float(total - free) / (1024 * 1024) << " MB\n"
		 << "free mem:  " << float(free) / (1024 * 1024)  << " MB\n"
		 << "total mem: " << float(total) / (1024 * 1024) << " MB\n\n";

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_fo);
	cudaFree(dev_u);
	cudaFree(dev_g_bottom);
	cudaFree(dev_g_right);
	cudaFree(dev_g_top);
	cudaFree(dev_g_left);
	cudaFree(dev_rr_bottom);
	cudaFree(dev_rr_right);
	cudaFree(dev_rr_top);
	cudaFree(dev_rr_left);
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


	for(p = 0; p < 160; p++)
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
		/* int offset = i + NX * j; */
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

		getn(fo, i, j) = value;
		/* fo(i, j) = value; */


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

/* __global__ void propagation(int jp1, int jp2, int ip1, int ip2, const float *f, float *u, int k) */
/* { */
/* 	// Map from threadIdx / BlockIdx to pixel position */

/* 	int i = threadIdx.x + blockIdx.x * blockDim.x; */
/* 	int j = threadIdx.y + blockIdx.y * blockDim.y; */

/* 	if(i < NX && j < NY) { */
/* 		float v = 1500.f * sqrtf(1.f + get(f, i, j)); */
/* 		float r = v * dt / hx; */
/* 		float s = 2.f - 4.f * r * r; */

/* 		float val; // will hold new u at (i, j, k + 1) */

/* 		// not at boundary */
/* 		if (i != 0 && i != NX - 1 && j != 0 && j != NY - 1) { */

/* 			val = */
/* 				r * r * */
/* 				(get(u, i+1, j, k) + */
/* 				 get(u, i-1, j, k) + */
/* 				 get(u, i, j-1, k) + */
/* 				 get(u, i, j+1, k)) + */
/* 				s * get(u, i, j, k) - */
/* 				get(u, i, j, k-1); */

/* 			// at sensor, k <= 24 */
/* 			if (j + 1 >= jp1 && j + 1 <= jp2 && i + 1 >= ip1 && i + 1 <= ip2 && k + 1 <= 24) { */
/* 				float t = k * dt - tt; */

/* 				// add wave value */
/* 				val += */
/* 					v * v * dt * dt * */
/* 					cosf(omegac * t) * */
/* 					expf(-(t * t) / (2.f * tao * tao)); */
/* 			} */
/* 		} */

/* 		// at boundary */
/* 		else { */

/* 			// index variables for different boundary cases */
/* 			// TODO: need better names */
/* 			int i_A, i_B, j_A, j_B; */

/* 			// top boundary */
/* 			if (j == 0) */
/* 			{ */
/* 				i_A = i; */
/* 				i_B = i; */
/* 				j_A = j + 1; */
/* 				j_B = j + 2; */
/* 			} */

/* 			// bottom boundary */
/* 			else if (j == NY - 1) */
/* 			{ */
/* 				i_A = i; */
/* 				i_B = i; */
/* 				j_A = j - 1; */
/* 				j_B = j - 2; */
/* 			} */

/* 			// left boundary */
/* 			else if (i == 0) */
/* 			{ */
/* 				i_A = i + 1; */
/* 				i_B = i + 2; */
/* 				j_A = j; */
/* 				j_B = j; */
/* 			} */

/* 			// right boundary */
/* 			else */
/* 			{ */
/* 				i_A = i - 1; */
/* 				i_B = i - 2; */
/* 				j_A = j; */
/* 				j_B = j; */
/* 			} */

/* 			val = */
/* 				(2.f - 2.f * r - r * r) * get(u, i, j, k) + */
/* 				2.f * r * (1.f + r) * get(u, i_A, j_A, k) - */
/* 				r * r * get(u, i_B, j_B, k) + */
/* 				(2.f * r - 1.f) * get(u, i, j, k-1) - */
/* 				2.f * r * get(u, i_A, j_A, k-1); */
/* 		} */

/* 		get(u, i, j, k+1) = val; */
/* 	} */
/* } */

__global__ void propagation(int jp1, int jp2, int ip1, int ip2, const float *f, float *u, int k, int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i < NX && j < NY) {
		float v = 1500.f * sqrtf(1.f + getn(f, i, j));
		float r = v * dt / hx;
		float s = 2.f - 4.f * r * r;

		float val; // will hold new u at (i, j, k + 1)

		// not at boundary
		if (i != 0 && i != NX - 1 && j != 0 && j != NY - 1) {

			val =
				r * r *
				(getn(u, i+1, j, k, g) +
				 getn(u, i-1, j, k, g) +
				 getn(u, i, j-1, k, g) +
				 getn(u, i, j+1, k, g)) +
				s * getn(u, i, j, k, g) -
				getn(u, i, j, k-1, g);

			// at sensor, k <= 24
			if (j + 1 >= jp1 && j + 1 <= jp2 && i + 1 >= ip1 && i + 1 <= ip2 && k + 1 <= 24) {
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

			// top boundary
			if (j == 0)
			{
				i_A = i;
				i_B = i;
				j_A = j + 1;
				j_B = j + 2;
			}

			// bottom boundary
			else if (j == NY - 1)
			{
				i_A = i;
				i_B = i;
				j_A = j - 1;
				j_B = j - 2;
			}

			// left boundary
			else if (i == 0)
			{
				i_A = i + 1;
				i_B = i + 2;
				j_A = j;
				j_B = j;
			}

			// right boundary
			else
			{
				i_A = i - 1;
				i_B = i - 2;
				j_A = j;
				j_B = j;
			}

			val =
				(2.f - 2.f * r - r * r) * getn(u, i, j, k, g) +
				2.f * r * (1.f + r) * getn(u, i_A, j_A, k, g) -
				r * r * getn(u, i_B, j_B, k, g) +
				(2.f * r - 1.f) * getn(u, i, j, k-1, g) -
				2.f * r * getn(u, i_A, j_A, k-1, g);
		}

		getn(u, i, j, k+1, g) = val;

		/* if (k+1 == NT - 1) */
			/* printf("%e \t", getn(u, i, j, k+1, g)); */
	}
}

/* __global__ void propagation_at_corners(float *u) */
/* { */
/* 	int k = threadIdx.x + blockIdx.x * blockDim.x; */

/* 	if (k < NT) { */
/* 		get(u, 0, 0, k) = */
/* 			1.f / 2.f * (get(u, 0, 1, k) + get(u, 1, 0, k)); */

/* 		get(u, NX-1, 0, k) = */
/* 			1.f / 2.f * (get(u, NX-2, 0, k) + get(u, NX-1, 1, k)); */

/* 		get(u, 0, NY-1, k) = */
/* 			1.f / 2.f * (get(u, 0, NY-2, k) + get(u, 1, NY-1, k)); */

/* 		get(u, NX-1, NY-1, k) = */
/* 			1.f / 2.f * (get(u, NX-2, NY-1, k) + get(u, NX-1, NY-2, k)); */
/* 	} */
/* } */

__global__ void propagation_at_corners(float *u, int g)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < NT) {
		getn(u, 0, 0, k, g) =
			1.f / 2.f * (getn(u, 0, 1, k, g) + getn(u, 1, 0, k, g));

		getn(u, NX-1, 0, k, g) =
			1.f / 2.f * (getn(u, NX-2, 0, k, g) + getn(u, NX-1, 1, k, g));

		getn(u, 0, NY-1, k, g) =
			1.f / 2.f * (getn(u, 0, NY-2, k, g) + getn(u, 1, NY-1, k, g));

		getn(u, NX-1, NY-1, k, g) =
			1.f / 2.f * (getn(u, NX-2, NY-1, k, g) + getn(u, NX-1, NY-2, k, g));
	}
}

__global__ void initial_signal(const float *u, float *g_bottom, float *g_right, float *g_top, float *g_left, int g)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store values at bottom sensor row of u
	getn(g_bottom, i+21, k+2, g, 0, NX, NT) =
		getn(u, i+21, 180, k+2, g);


	// store values at top sensor row of u
	getn(g_top, i+21, k+2, g, 0, NX, NT) =
		getn(u, i+21, 20, k+2, g);


	// store values at right sensor column of u
	getn(g_right, i+21, k+2, g, 0, NX, NT) =
		getn(u, 180, i+21, k+2, g);

	// store values at left sensor column of u
	getn(g_left, i+21, k+2, g, 0, NX, NT) =
		getn(u, 20, i+21, k+2, g);

	/* printf("%e \t", getn(u, 20, i+21, k+2, g)); */
}

__global__ void difference_signal(const float *u, const float *g_bottom, const float *g_right, const float *g_top, const float *g_left, float *rr_bottom, float *rr_right, float *rr_top, float *rr_left, int g)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store difference at time k+2 of original signal
	// and current signal at bottom sensor row
	getn(rr_bottom, i+21, k+2, g, 0, NX, NT) =
		getn(g_bottom, i+21, k+2, g, 0, NX, NT) -
		getn(u, i+21, 180, k+2, g);

	/* printf("%e \t", getn(rr_bottom, i+21, k+2, g, 0, NX, NT)); */

	// store difference at time k+2 of original signal
	// and current signal at top sensor row
	getn(rr_top, i+21, k+2, g, 0, NX, NT) =
		getn(g_top, i+21, k+2, g, 0, NX, NT) -
		getn(u, i+21, 20, k+2, g);

	// store difference at time k+2 of original signal
	// and current signal at right sensor column
	getn(rr_right, i+21, k+2, g, 0, NX, NT) =
		getn(g_right, i+21, k+2, g, 0, NX, NT) -
		getn(u, 180, i+21, k+2, g);

	// store difference at time k+2 of original signal
	// and current signal at left sensor column
	getn(rr_left, i+21, k+2, g, 0, NX, NT) =
		getn(g_left, i+21, k+2, g, 0, NX, NT) -
		getn(u, 20, i+21, k+2, g);
}

__global__ void backpropagation1(float *z, const float *f, int k, int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= 1 && i < (NX - 1) && j >= 1 && j < (NY - 1))
	{
		getn(z, i, j, k, g, NX, NY, NT+1) =
			1500.f * 1500.f * (dt * dt) *
			((1.f + getn(f, i, j-1)) * getn(z, i, j-1, k+1, g, NX, NY, NT+1) +
			 (1.f + getn(f, i, j+1)) * getn(z, i, j+1, k+1, g, NX, NY, NT+1) +
			 (1.f + getn(f, i-1, j)) * getn(z, i-1, j, k+1, g, NX, NY, NT+1) +
			 (1.f + getn(f, i+1, j)) * getn(z, i+1, j, k+1, g, NX, NY, NT+1) -
			 4.f * (1.f + getn(f, i, j)) *
			 getn(z, i, j, k+1, g, NX, NY, NT+1)) / (h * h) +
			2.f * getn(z, i, j, k+1, g, NX, NY, NT+1) -
			getn(z, i, j, k+2, g, NX, NY, NT+1);

		/* if (k == 1) */
			/* printf("%e \t", getn(z, i, j, k, g, NX, NY, NT+1)); */
	}
}

__global__ void backpropagation2(float *z, const float *rr_bottom, const float *rr_right, const float *rr_top, const float *rr_left, int k, int g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= 21 && i < 180) {
		getn(z, i, 180, k, g, NX, NY, NT+1) =
			getn(z, i, 179, k, g, NX, NY, NT+1) +
			getn(rr_bottom, i, k) * h * 1000.f;

		getn(z, i, 20, k, g, NX, NY, NT+1) =
			getn(z, i, 21, k, g, NX, NY, NT+1) +
			getn(rr_top, i, k, g, 0, NX, NT) * h * 1000.f;

		getn(z, 180, i, k, g, NX, NY, NT+1) =
			getn(z, 179, i, k, g, NX, NY, NT+1) +
			getn(rr_right, i, k, g, 0, NX, NT) * h * 1000.f;

		getn(z, 20, i, k, g, NX, NY, NT+1) =
			getn(z, 21, i, k, g, NX, NY, NT+1) +
			getn(rr_left, i, k, g, 0, NX, NT) * h * 1000.f;
	}

	if (i >= 1 && i < (NX - 1)) {
		getn(z, i, 0, k, g, NX, NY, NT+1) =
			getn(z, i, 1, k, g, NX, NY, NT+1);

		getn(z, i, NY-1, k, g, NX, NY, NT+1) =
			getn(z, i, NY-2, k, g, NX, NY, NT+1);

		getn(z, 0, i, k, g, NX, NY, NT+1) =
			getn(z, 1, i, k, g, NX, NY, NT+1);

		getn(z, NX-1, i, k, g, NX, NY, NT+1) =
			getn(z, NX-2, i, k, g, NX, NY, NT+1);
	}

	else if (i == 0) {
		getn(z, 0, 0, k, g, NX, NY, NT+1) =
			(getn(z, 1, 0, k, g, NX, NY, NT+1) +
			 getn(z, 0, 1, k, g, NX, NY, NT+1)) / 2.f;

		getn(z, NX-1, 0, k, g, NX, NY, NT+1) =
			(getn(z, NX-2, 0, k, g, NX, NY, NT+1) +
			 getn(z, NX-1, 1, k, g, NX, NY, NT+1)) / 2.f;

		getn(z, 0, NY-1, k) =
			(getn(z, 1, NY-1, k, g, NX, NY, NT+1) +
			 getn(z, 0, NY-2, k, g, NX, NY, NT+1)) / 2.f;

		getn(z, NX-1, NY-1, k, g, NX, NY, NT+1) =
			(getn(z, NX-2, NY-1, k, g, NX, NY, NT+1) +
			 getn(z, NX-1, NY-2, k, g, NX, NY, NT+1)) / 2.f;
	}
}

__global__ void laplace(const float *u, float *Lu, int g)
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

		getn(Lu, i, j, k+1, g) =
			(getn(u, i, j_prev, k+1, g) +
			 getn(u, i, j_next, k+1, g) +
			 getn(u, i_prev, j, k+1, g) +
			 getn(u, i_next, j, k+1, g) -
			 4.f * getn(u, i, j, k+1, g)) / (h * h);
	}
}

__global__ void laplace_corners(const float *u, float *Lu, int g)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if ((k + 1) < NT) {
		getn(Lu, 0, 0, k+1, g) =
			(getn(Lu, 1, 0, k+1, g) +
			 getn(Lu, 0, 1, k+1, g)) / 2.f;

		getn(Lu, NX-1, 0, k+1, g) =
			(getn(Lu, NX-2, 0, k+1, g) +
			 getn(Lu, NX-1, 1, k+1, g)) / 2.f;

		getn(Lu, 0, NY-1, k+1, g) =
			(getn(Lu, 1, NY-1, k+1, g) +
			 getn(Lu, 0, NY-2, k+1, g)) / 2.f;

		getn(Lu, NX-1, NY-1, k+1, g) =
			(getn(Lu, NX-2, NY-1, k+1, g) +
			 getn(Lu, NX-1, NY-2, k+1, g)) / 2.f;
	}
}

__global__ void update_differential(float *df, const float *z, const float *Lu, const float *f, int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < NX && j < NY && (k + 1) < NT) {

		atomicAdd(
			&getn(df, i, j),
			getn(z, i, j, k+1, g, NX, NY, NT+1) *
			getn(Lu, i, j, k+1, g) /
			(1.f + getn(f, i, j)));
	}
}

__global__ void print_differential(const float *df)
{
	for (int j = 0; j < NY; ++j) {
		for (int i = 0; i < NX; ++i) {
			printf("%f \t", getn(df, i, j));
		}
		printf("\n");
	}
}

__global__ void update_field(float *f, const float *df, float *f_minus_fo, const float *fo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < NX && j < NY)
	{
		bool in_sensor_field = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

		float alpha = in_sensor_field ? 1.f : 0.f;

		getn(f, i, j) += 20000.f * alpha * getn(df, i, j);

		getn(f_minus_fo, i, j) = getn(f, i, j) - getn(fo, i, j);
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

	for(i = 0; i < NX; i++) {
		x_file << x[i];
		x_file << "\n";
	}

	for(j = 0; j < NX; j++) {
		y_file << y[j];
		y_file << "\n";
	}

	for(j = 0; j < NY; j++) {
		for(i = 0; i < NX; i++) {
			fo_file << getn(fo, i, j);
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(j = 0; j < NY; j++) {
		for(i = 0; i < NX; i++) {
			f_file << getn(f, i, j);
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
			sum += getn(A, i, j) * getn(A, i, j);

	return sqrtf(sum);
}
