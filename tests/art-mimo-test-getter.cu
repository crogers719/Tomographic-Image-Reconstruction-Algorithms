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

#define NS 640 							//number of sensors

#define BLOCK_X 16
#define BLOCK_Y 16

template <typename T, int Nx>
struct Get2d
{
	__device__
	Get2d(T* _ptr)
		: ptr(_ptr)
	{}

	__device__
	T& operator()(int i, int j)
	{
		return ptr[i + Nx * j];
	}

	__device__
	const T& operator()(int i, int j) const
	{
		return ptr[i + Nx * j];
	}

	T* ptr;
};

template <typename T, int Nx, int Ny>
struct Get3d
{
	__device__
	Get3d(T* _ptr)
		: ptr(_ptr)
	{}

	__device__
	T& operator()(int i, int j, int k)
	{
		return ptr[i + Nx * j + Nx * Ny * k];
	}

	__device__
	const T& operator()(int i, int j, int k) const
	{
		return ptr[i + Nx * j + Nx * Ny * k];
	}

	T* ptr;
};

template <typename T, int Nx, int Ny, int Nz>
struct Get4d
{
	__device__
	Get4d(T* _ptr)
		: ptr(_ptr)
	{}

	__device__
	T& operator()(int i, int j, int k, int l)
	{
		return ptr[i + Nx * j + Nx * Ny * k + Nx * Ny * Nz * l];
	}

	__device__
	const T& operator()(int i, int j, int k, int l) const
	{
		return ptr[i + Nx * j + Nx * Ny * k + Nx * Ny * Nz * l];
	}

	T* ptr;
};

__constant__ float hx = 0.001f;
__constant__ float hy = 0.001f; // pixel size
__constant__ float h = 0.001f;

/* __constant__ float T = 1.3333e-04f; // 0.2f / 1500.f; */
__constant__ float dt = 3.3333e-07f; // T / 400.f;
/* __constant__ float fre = 125000.f; */
__constant__ float omegac = 7.8540e+05f; // 2.f * pi * fre; // wavelength
__constant__ float tao = 4.0000e-06f; // pi / omegac;
__constant__ float tt = 8.1573e-06f; // sqrtf(6.f * logf(2.f)) * tao; // time  delay

// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti);
void Position_Transducers (int *&, int *&, int);
void IO_Files(float*, float*, float*, float*);
float norm(float*, int, int);

template <typename T>
__host__ __device__
T& get(T* ptr, int i, int j = 0, int k = 0, int nx = NX, int ny = NY)
{
	return ptr[i + nx * j + nx * ny * k];
}

__global__ void field_setup(const float *x, const float *y, Get2d<float, NX> fo)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		float value = 0.f;

		float rc = 0.015f;
		float rp = 0.005f;

		float sc = 0.03f;
		float sp = 0.05f;

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

		fo(i, j) = value;
	}
}

__global__ void propagation(
		int jp1, int jp2, int ip1, int ip2, 
		const Get2d<float, NX> f,
		Get3d<float, NX, NY> u, 
		int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i < NX && j < NY) {
		float v = 1500.f * sqrtf(1.f + f(i, j));
		float r = v * dt / hx;
		float s = 2.f - 4.f * r * r;

		float val; // will hold new u at (i, j, k + 1)

		// not at boundary
		if (i != 0 && i != NX - 1 && j != 0 && j != NY - 1) {

			val =
				r * r *
				(u(i+1, j, k) +
				 u(i-1, j, k) +
				 u(i, j-1, k) +
				 u(i, j+1, k)) +
				s * u(i, j, k) -
				u(i, j, k-1);

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

			// boundary booleans
			bool top = (j == 0);
			bool bottom = (j == NY - 1);
			bool left = (i == 0);
			bool right = (i == NX - 1);

			// index variables for different boundary cases
			int ja = top ? (j + 1) : bottom ? (j - 1) : j;
			int jb = top ? (j + 2) : bottom ? (j - 2) : j;

			int ia = left ? (i + 1) : right ? (i - 1) : i;
			int ib = left ? (i + 2) : right ? (i - 2) : i;

			val =
				(2.f - 2.f * r - r * r) * u(i, j, k) +
				2.f * r * (1.f + r) * u(ia, ja, k) -
				r * r * u(ib, jb, k) +
				(2.f * r - 1.f) * u(i, j, k-1) -
				2.f * r * u(ia, ja, k-1);
		}

		u(i, j, k+1) = val;

		/* if (k+1 == NT - 1) */
			/* printf("%e \t", val); */
	}
}

__global__ void propagation_at_corners(Get3d<float, NX, NY> u)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < NT) {
		u(0, 0, k) =
			1.f / 2.f * (u(0, 1, k) + u(1, 0, k));

		u(NX-1, 0, k) =
			1.f / 2.f * (u(NX-2, 0, k) + u(NX-1, 1, k));

		u(0, NY-1, k) =
			1.f / 2.f * (u(0, NY-2, k) + u(1, NY-1, k));

		u(NX-1, NY-1, k) =
			1.f / 2.f * (u(NX-2, NY-1, k) + u(NX-1, NY-2, k));
	}
}

__global__ void initial_signal(
		const Get3d<float, NX, NY> u,
		Get3d<float, NX, NT> g_bottom,
		Get3d<float, NX, NT> g_right,
		Get3d<float, NX, NT> g_top,
		Get3d<float, NX, NT> g_left,
		int p)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store values at bottom sensor row of u
	g_bottom(i+21, k+2, p) =
		u(i+21, 180, k+2);

	// store values at top sensor row of u
	g_top(i+21, k+2, p) =
		u(i+21, 20, k+2);



	// store values at right sensor column of u
	g_right(i+21, k+2, p) =
		u(180, i+21, k+2);


	// store values at left sensor column of u
	g_left(i+21, k+2, p) =
		u(20, i+21, k+2);

	/* printf("%e \t", u(20, i+21, k+2)); */
}

__global__ void difference_signal(
		const Get3d<float, NX, NY> u,
		const Get3d<float, NX, NT> g_bottom,
		const Get3d<float, NX, NT> g_right,
		const Get3d<float, NX, NT> g_top,
		const Get3d<float, NX, NT> g_left,
		Get2d<float, NX> rr_bottom,
		Get2d<float, NX> rr_right,
		Get2d<float, NX> rr_top,
		Get2d<float, NX> rr_left,
		int p)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store difference at time k+2 of original signal
	// and current signal at bottom sensor row
	rr_bottom(i+21, k+2) =
		g_bottom(i+21, k+2, p) -
		u(i+21, 180, k+2);

	/* printf("%e ", rr_bottom(i+21, k+2)); */

	// store difference at time k+2 of original signal
	// and current signal at top sensor row
	rr_top(i+21, k+2) =
		g_top(i+21, k+2, p) -
		u(i+21, 20, k+2);

	// store difference at time k+2 of original signal
	// and current signal at right sensor column
	rr_right(i+21, k+2) =
		g_right(i+21, k+2, p) -
		u(180, i+21, k+2);

	// store difference at time k+2 of original signal
	// and current signal at left sensor column
	rr_left(i+21, k+2) =
		g_left(i+21, k+2, p) -
		u(20, i+21, k+2);
}

__global__ void backpropagation1(
		Get3d<float, NX, NY> z,
		const Get2d<float, NX> f,
		int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= 1 && i < (NX - 1) && j >= 1 && j < (NY - 1))
	{
		z(i, j, k) =
			1500.f * 1500.f * (dt * dt) *
			((1.f + f(i, j-1)) * z(i, j-1, k+1) +
			 (1.f + f(i, j+1)) * z(i, j+1, k+1) +
			 (1.f + f(i-1, j)) * z(i-1, j, k+1) +
			 (1.f + f(i+1, j)) * z(i+1, j, k+1) -
			 4.f * (1.f + f(i, j)) *
			 z(i, j, k+1)) / (h * h) +
			2.f * z(i, j, k+1) -
			z(i, j, k+2);

		/* if (k == 1) */
			/* printf("%e \t", z(i, j, k)); */
	}
}

__global__ void backpropagation2(
		Get3d<float, NX, NY> z,
		const Get2d<float, NX> rr_bottom,
		const Get2d<float, NX> rr_right,
		const Get2d<float, NX> rr_top,
		const Get2d<float, NX> rr_left,
		int k)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= 21 && i < 180) {
		z(i, 180, k) =
			z(i, 179, k) +
			rr_bottom(i, k) * h * 1000.f;

		z(i, 20, k) =
			z(i, 21, k) +
			rr_top(i, k) * h * 1000.f;

		z(180, i, k) =
			z(179, i, k) +
			rr_right(i, k) * h * 1000.f;

		z(20, i, k) =
			z(21, i, k) +
			rr_left(i, k) * h * 1000.f;
	}

	if (i >= 1 && i < (NX - 1)) {
		z(i, 0, k) =
			z(i, 1, k);

		z(i, NY-1, k) =
			z(i, NY-2, k);

		z(0, i, k) =
			z(1, i, k);

		z(NX-1, i, k) =
			z(NX-2, i, k);
	}

	else if (i == 0) {
		z(0, 0, k) =
			(z(1, 0, k) +
			 z(0, 1, k)) / 2.f;

		z(NX-1, 0, k) =
			(z(NX-2, 0, k) +
			 z(NX-1, 1, k)) / 2.f;

		z(0, NY-1, k) =
			(z(1, NY-1, k) +
			 z(0, NY-2, k)) / 2.f;

		z(NX-1, NY-1, k) =
			(z(NX-2, NY-1, k) +
			 z(NX-1, NY-2, k)) / 2.f;
	}
}

__global__ void laplace(
		const Get3d<float, NX, NY> u,
		Get3d<float, NX, NY> Lu)
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

		Lu(i, j, k+1) =
			(u(i, j_prev, k+1) +
			 u(i, j_next, k+1) +
			 u(i_prev, j, k+1) +
			 u(i_next, j, k+1) -
			 4.f * u(i, j, k+1)) / (h * h);
	}
}

__global__ void laplace_corners(const Get3d<float, NX, NY> u, Get3d<float, NX, NY> Lu)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if ((k + 1) < NT) {
		Lu(0, 0, k+1) =
			(Lu(1, 0, k+1) +
			 Lu(0, 1, k+1)) / 2.f;

		Lu(NX-1, 0, k+1) =
			(Lu(NX-2, 0, k+1) +
			 Lu(NX-1, 1, k+1)) / 2.f;

		Lu(0, NY-1, k+1) =
			(Lu(1, NY-1, k+1) +
			 Lu(0, NY-2, k+1)) / 2.f;

		Lu(NX-1, NY-1, k+1) =
			(Lu(NX-2, NY-1, k+1) +
			 Lu(NX-1, NY-2, k+1)) / 2.f;
	}
}

__global__ void update_differential(
		Get2d<float, NX> df,
		Get3d<float, NX, NY> const z,
		Get3d<float, NX, NY> const Lu,
		Get2d<float, NX> const f)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < NX && j < NY && (k + 1) < NT) {

		atomicAdd(
			&df(i, j),
			z(i, j, k+1) *
			Lu(i, j, k+1) /
			(1.f + f(i, j)));
	}
}

__global__ void update_field(
		Get2d<float, NX> f,
		Get2d<float, NX> const df,
		Get2d<float, NX> f_minus_fo,
		Get2d<float, NX> const fo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < NX && j < NY)
	{
		bool in_sensor_field = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

		float alpha = in_sensor_field ? 1.f : 0.f;

		f(i, j) += 20000.f * alpha * df(i, j);
		f_minus_fo(i, j) = f(i, j) - fo(i, j);
	}
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

	// set floting-point precision on stdout and stderr
	cout << fixed << setprecision(10);
	cerr << fixed << setprecision(10);

	cerr << "Ultrasonic Tomography Running:\n\n";

	ti = clock();
	cerr << "ti = " << ti << "\n";

	Ultrasonic_Tomography(group_size, target_epsilon, max_iterations, ti);

	tf = clock();
	cerr << "tf = " << tf << "\n"
		 << "tt = " << tf - ti << "\n"
		 << "Total Seconds = " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";

	// End of the program

	return 0;
}

inline int grid_size(int n, int threads)
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
	int size_u = NX * NY * NT * sizeof(float);


	float *dev_g_bottom;
	float *dev_g_right;
	float *dev_g_top;
	float *dev_g_left;
	int size_g = NX * NT * (NS / group_size) * sizeof(float);


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


	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
			grid_size(NX, threads_propagation.x),
			grid_size(NY, threads_propagation.y));

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

		propagation_at_corners<<<NT, 1>>>(dev_u);

		initial_signal<<<NT - 2, 159>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, p / group_size);
	}

	// Kaczmarz method
	// propagation

	float *dev_rr_bottom;
	int size_rr_bottom = NX * NT * sizeof(float);

	float *dev_rr_right;
	int size_rr_right = NX * NT * sizeof(float);

	float *dev_rr_top;
	int size_rr_top = NX * NT * sizeof(float);

	float *dev_rr_left;
	int size_rr_left = NX * NT * sizeof(float);

	float *dev_z;
	int size_z = NX * NY * (NT + 1) * sizeof(float);

	float *dev_Lu;
	int size_Lu = NX * NY * NT * sizeof(float);

	float *dev_f;
	int size_f = NX * NY * sizeof(float);

	float *dev_df;
	int size_df = NX * NY * sizeof(float);

	float *dev_f_minus_fo;
	int size_f_minus_fo = NX * NY * sizeof(float);

	// Allocation

	cudaMalloc((void**) &dev_rr_bottom, size_rr_bottom);
	cudaMalloc((void**) &dev_rr_right, size_rr_right);
	cudaMalloc((void**) &dev_rr_top, size_rr_top);
	cudaMalloc((void**) &dev_rr_left, size_rr_left);
	cudaMalloc((void**) &dev_z, size_z);
	cudaMalloc((void**) &dev_Lu, size_Lu);
	cudaMalloc((void**) &dev_f, size_f);
	cudaMalloc((void**) &dev_df, size_df);
	cudaMalloc((void**) &dev_f_minus_fo, size_f_minus_fo);

	cudaMemset(dev_rr_bottom, 0.f, size_rr_bottom);
	cudaMemset(dev_rr_right, 0.f, size_rr_right);
	cudaMemset(dev_rr_top, 0.f, size_rr_top);
	cudaMemset(dev_rr_left, 0.f, size_rr_left);
	cudaMemset(dev_f, 0.f, size_f);
	cudaMemset(dev_Lu, 0.f, size_Lu);

	float *f = new float[NX * NY];
	float *f_minus_fo = new float[NX * NY];

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'art_convergence.txt'...\n"
		 << "writing time to 'art_time.txt'...\n";

	ofstream convergence_file("art_convergence.txt");
	ofstream time_file("art_time.txt");

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

			propagation_at_corners<<<NT, 1>>>(dev_u);
			difference_signal<<<NT - 2, 159>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, p / group_size);

			cudaMemset(dev_z, 0.f, size_z);

			for(k = NT - 2; k > 0; k--)
			{
				backpropagation1<<<grid_backpropagation1, threads_backpropagation1>>>(dev_z, dev_f, k);
				backpropagation2<<<NX, 1>>>(dev_z, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, k);
			}

			laplace<<<grid_laplace, threads_laplace>>>(dev_u, dev_Lu);
			laplace_corners<<<NT, 1>>>(dev_u, dev_Lu);

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

	cout << endl;

	cudaMemcpy(f, dev_f, size_f, cudaMemcpyDeviceToHost);

	IO_Files(x, y, fo, f);

	// Free Variables

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
			fo_file << get(fo, i, j);
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(j = 0; j < NY; j++) {
		for(i = 0; i < NX; i++) {
			f_file << get(f, i, j);
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
			sum += get(A, i, j) * get(A, i, j);

	return sqrtf(sum);
}
