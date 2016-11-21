// HEADERS

#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>

#include "cuda_ptr.cuh"
using namespace std;

// DEFINITIONS

#define NX 192				//was 201
#define NY 192				//was 201
#define NT 401

#define NS 640 				//number of sensors

#define BLOCK_X 16
#define BLOCK_Y 16
#define hx 0.001f
#define hy 0.001f
#define h 0.001f
#define dt 3.3333e-07f
#define omegac 7.8540e+05f
#define tao 4.0000e-06f
#define tt 8.1573



// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti);

void Position_Transducers(host_ptr<int> ii, host_ptr<int> jj, int num)
{
	//returns the (x,y) coordinates of the number of total transducers
	int p = 0;

	for(p = 0; p < 160; p++) {
		ii(p) = 21 + (p + 1);
		jj(p) = 181;
	}

	for(p = 160; p < 320; p++) {
		ii(p) = 181;
		jj(p) = 181 - ((p + 1) - 160);
	}

	for(p = 320; p < 480; p++) {
		ii(p) = 181 - ((p + 1) - 320);
		jj(p) = 21;
	}

	for(p = 480; p < num; p++) {
		ii(p) = 21;
		jj(p) = 21 + ((p + 1) - 480);
	}
}

template <int Nx, int Ny>
void IO_Files(host_ptr<float> const x, host_ptr<float> const y, host_ptr<float> const fo, host_ptr<float> const f)
{
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

	for(int i = 0; i < Nx; i++) {
		x_file << x(i);
		x_file << "\n";
	}

	for(int j = 0; j < Nx; j++) {
		y_file << y(j);
		y_file << "\n";
	}

	for(int j = 0; j < Ny; j++) {
		for(int i = 0; i < Nx; i++) {
			fo_file << fo(i, j);
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(int j = 0; j < Ny; j++) {
		for(int i = 0; i < Nx; i++) {
			f_file << f(i, j);
			f_file << " ";
		}

		f_file << "\n";
	}

	x_file.close();
	y_file.close();
	fo_file.close();
	f_file.close();
}

template <int Nx, int Ny>
float norm(host_ptr<float> const A)
{
	float sum = 0;

	for (int j = 0; j < Ny; ++j)
		for (int i = 0; i < Nx; ++i)
			sum += A(i, j) * A(i, j);

	return sqrtf(sum);
}

__host__ __device__
inline int grid_size(int n, int threads)
{
	return ceil((float)(n) / threads);
}

__global__ void field_setup(kernel_ptr<float> const x, kernel_ptr<float> const y, kernel_ptr<float> fo)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY)) {
		float value = 0.f;

		float rc = 0.015f;
		float rp = 0.005f;

		float sc = 0.03f;
		float sp = 0.05f;

		if (powf(x(i), 2) + powf(y(j), 2) <= powf(rc, 2)) {
			value = sc;
		}

		else if (powf(x(i) - rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y(j) - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2)) {
			value = sp;
		}

		else if (powf(x(i) + rc * cos(-30 * (3.14159265f / 180)), 2) + powf(y(j) - rc * sin(30 * (3.14159265f / 180)), 2) <= powf(rp, 2)) {
			value = sp;
		}

		else if (powf(x(i), 2) + powf(y(j) + rc, 2) <= powf(rp, 2)) {
			value = sp;
		}

		fo(i, j) = value;
	}
}

__global__
void propagation(
		int jp1, int jp2, int ip1, int ip2,
		kernel_ptr<float> const f,
		kernel_ptr<float>u,
		int k, int g)
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
				(u(i+1, j, k, g) +
				 u(i-1, j, k, g) +
				 u(i, j-1, k, g) +
				 u(i, j+1, k, g)) +
				s * u(i, j, k, g) -
				u(i, j, k-1, g);

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
			int ia = left ? i + 1 : right ? i - 1 : i;
			int ib = left ? i + 2 : right ? i - 2 : i;

			int ja = top ? j + 1 : bottom ? j - 1 : j;
			int jb = top ? j + 2 : bottom ? j - 2 : j;

			val =
				(2.f - 2.f * r - r * r) * u(i, j, k, g) +
				2.f * r * (1.f + r) * u(ia, ja, k, g) -
				r * r * u(ib, jb, k, g) +
				(2.f * r - 1.f) * u(i, j, k-1, g) -
				2.f * r * u(ia, ja, k-1, g);
		}

		u(i, j, k+1, g) = val;

		/* if (k+1 == NT - 1) */
			/* printf("%e \t", u(i, j, k+1, g)); */
	}
}

__global__ void propagation_at_corners(kernel_ptr<float> u, int g)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if (k < NT) {
		u(0, 0, k, g) =
			1.f / 2.f * (u(0, 1, k, g) + u(1, 0, k, g));

		u(NX-1, 0, k, g) =
			1.f / 2.f * (u(NX-2, 0, k, g) + u(NX-1, 1, k, g));

		u(0, NY-1, k, g) =
			1.f / 2.f * (u(0, NY-2, k, g) + u(1, NY-1, k, g));

		u(NX-1, NY-1, k, g) =
			1.f / 2.f * (u(NX-2, NY-1, k, g) + u(NX-1, NY-2, k, g));
	}
}

__global__ void initial_signal(
		kernel_ptr<float> const u,
		kernel_ptr<float> g_bottom,
		kernel_ptr<float> g_right,
		kernel_ptr<float> g_top,
		kernel_ptr<float> g_left,
		int g)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store values at bottom sensor row of u
	g_bottom(i+21, k+2, g) = u(i+21, 180, k+2, g);


	// store values at top sensor row of u
	g_top(i+21, k+2, g) = u(i+21, 20, k+2, g);


	// store values at right sensor column of u
	g_right(i+21, k+2, g) = u(180, i+21, k+2, g);

	// store values at left sensor column of u
	g_left(i+21, k+2, g) = u(20, i+21, k+2, g);

	/* printf("%e \t", u(20, i+21, k+2, g)); */
}

__global__ void difference_signal(
		kernel_ptr<float> const u,
		kernel_ptr<float> const g_bottom,
		kernel_ptr<float> const g_right,
		kernel_ptr<float> const g_top,
		kernel_ptr<float> const g_left,
		kernel_ptr<float> rr_bottom,
		kernel_ptr<float> rr_right,
		kernel_ptr<float> rr_top,
		kernel_ptr<float> rr_left,
		int g)
{
	int i = threadIdx.x;
	int k = blockIdx.x;

	// store difference at time k+2 of original signal
	// and current signal at bottom sensor row
	rr_bottom(i+21, k+2, g) =
		g_bottom(i+21, k+2, g) -
		u(i+21, 180, k+2, g);

	/* printf("%e \t", rr_bottom(i+21, k+2, g, 0, NX, NT)); */

	// store difference at time k+2 of original signal
	// and current signal at top sensor row
	rr_top(i+21, k+2, g) =
		g_top(i+21, k+2, g) -
		u(i+21, 20, k+2, g);

	// store difference at time k+2 of original signal
	// and current signal at right sensor column
	rr_right(i+21, k+2, g) =
		g_right(i+21, k+2, g) -
		u(180, i+21, k+2, g);

	// store difference at time k+2 of original signal
	// and current signal at left sensor column
	rr_left(i+21, k+2, g) =
		g_left(i+21, k+2, g) -
		u(20, i+21, k+2, g);
}

__global__ void backpropagation1(
		kernel_ptr<float> z,
		kernel_ptr<float> const f,
		int k, int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= 1 && i < (NX - 1) && j >= 1 && j < (NY - 1)) {
		z(i, j, k, g) =
			1500.f * 1500.f * (dt * dt) *
			((1.f + f(i, j-1)) * z(i, j-1, k+1, g) +
			 (1.f + f(i, j+1)) * z(i, j+1, k+1, g) +
			 (1.f + f(i-1, j)) * z(i-1, j, k+1, g) +
			 (1.f + f(i+1, j)) * z(i+1, j, k+1, g) -
			 4.f * (1.f + f(i, j)) *
			 z(i, j, k+1, g)) / (h * h) +
			2.f * z(i, j, k+1, g) -
			z(i, j, k+2, g);		
	}
}

__global__ void backpropagation2(
		kernel_ptr<float> z,
		kernel_ptr<float> const rr_bottom,
		kernel_ptr<float> const rr_right,
		kernel_ptr<float> const rr_top,
		kernel_ptr<float> const rr_left,
		int k, int g)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= 21 && i < 180) {
		z(i, 180, k, g) =
			z(i, 179, k, g) + rr_bottom(i, k, g) * h * 1000.f;

		z(i, 20, k, g) =
			z(i, 21, k, g) + rr_top(i, k, g) * h * 1000.f;

		z(180, i, k, g) =
			z(179, i, k, g) + rr_right(i, k, g) * h * 1000.f;

		z(20, i, k, g) =
			z(21, i, k, g) + rr_left(i, k, g) * h * 1000.f;
	}

	if (i >= 1 && i < (NX - 1)) {
		z(i, 0, k, g) = z(i, 1, k, g);

		z(i, NY-1, k, g) = z(i, NY-2, k, g);

		z(0, i, k, g) = z(1, i, k, g);

		z(NX-1, i, k, g) = z(NX-2, i, k, g);
	}

	else if (i == 0) {
		z(0, 0, k, g) =
			(z(1, 0, k, g) + z(0, 1, k, g)) / 2.f;

		z(NX-1, 0, k, g) =
			(z(NX-2, 0, k, g) + z(NX-1, 1, k, g)) / 2.f;

		z(0, NY-1, k, g) =
			(z(1, NY-1, k, g) + z(0, NY-2, k, g)) / 2.f;

		z(NX-1, NY-1, k, g) =
			(z(NX-2, NY-1, k, g) + z(NX-1, NY-2, k, g)) / 2.f;
	}
}

__global__ void laplace(
		kernel_ptr<float> const u,
		kernel_ptr<float> Lu,
		int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i < NX && j < NY && (k + 1) < NT) {

		int ja = (j > 0) ? j - 1 : j;
		int jb = (j < NY - 1) ? j + 1 : j;

		int ia = (i > 0) ? i - 1 : i;
		int ib = (i < NX - 1) ? i + 1 : i;

		Lu(i, j, k+1, g) =
			(u(i, ja, k+1, g) +
			 u(i, jb, k+1, g) +
			 u(ia, j, k+1, g) +
			 u(ib, j, k+1, g) -
			 4.f * u(i, j, k+1, g)) / (h * h);
	}
}

__global__ void laplace_corners(
		kernel_ptr<float> const u,
		kernel_ptr<float> Lu,
		int g)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if ((k + 1) < NT) {
		Lu(0, 0, k+1, g) =
			(Lu(1, 0, k+1, g) +
			 Lu(0, 1, k+1, g)) / 2.f;

		Lu(NX-1, 0, k+1, g) =
			(Lu(NX-2, 0, k+1, g) +
			 Lu(NX-1, 1, k+1, g)) / 2.f;

		Lu(0, NY-1, k+1, g) =
			(Lu(1, NY-1, k+1, g) +
			 Lu(0, NY-2, k+1, g)) / 2.f;

		Lu(NX-1, NY-1, k+1, g) =
			(Lu(NX-2, NY-1, k+1, g) +
			 Lu(NX-1, NY-2, k+1, g)) / 2.f;
	}
}

__global__ void update_differential(
		kernel_ptr<float> df,
		kernel_ptr<float> const z,
		kernel_ptr<float> const Lu,
		kernel_ptr<float> const f,
		int g)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < NX && j < NY && (k + 1) < NT) {

		atomicAdd(
			&df(i, j),
			z(i, j, k+1, g) *
			Lu(i, j, k+1, g) /
			(1.f + f(i, j)));
	}
}

__global__ void update_field(
		kernel_ptr<float> f,
		kernel_ptr<float> const df,
		kernel_ptr<float> f_minus_fo,
		kernel_ptr<float> const fo,
		int Ng)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < NX && j < NY) {
		bool in_sensor_field = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

		float alpha = in_sensor_field ? 1.f : 0.f;

		f(i, j) += 20000.f * alpha * df(i, j)/(float)Ng;

		f_minus_fo(i, j) = f(i, j) - fo(i, j);
	}
}

__global__ void pre_sirt(
		const int group_size, const int num_groups,
		kernel_ptr<int> const ii,
		kernel_ptr<int> const jj,
		kernel_ptr<float> const fo,
		kernel_ptr<float> u,
		kernel_ptr<float> g_bottom,
		kernel_ptr<float> g_top,
		kernel_ptr<float> g_left,
		kernel_ptr<float> g_right
		)
{
	// unique id for each group
	int g = threadIdx.x + blockIdx.x * blockDim.x;

	if (g >= num_groups)
		return;

	int p = g * group_size;

	int jp1 = jj(p);
	int jp2 = jj(p + group_size - 1);
	int ip1 = ii(p);
	int ip2 = ii(p + group_size - 1);

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
		kernel_ptr<int> const ii,
		kernel_ptr<int> const jj,
		kernel_ptr<float> f,
		kernel_ptr<float> const fo,
		kernel_ptr<float> u,
		kernel_ptr<float> const g_bottom,
		kernel_ptr<float> const g_top,
		kernel_ptr<float> const g_left,
		kernel_ptr<float> const g_right,
		kernel_ptr<float> rr_bottom,
		kernel_ptr<float> rr_top,
		kernel_ptr<float> rr_left,
		kernel_ptr<float> rr_right,
		kernel_ptr<float> z,
		kernel_ptr<float> Lu,
		kernel_ptr<float> df,
		kernel_ptr<float> f_minus_fo
		)
{
	// unique id for each group
	int g = threadIdx.x + blockIdx.x * blockDim.x;

	if (g >= num_groups)
		return;

	int p = g * group_size;

	int jp1 = jj(p);
	int jp2 = jj(p + group_size - 1);
	int ip1 = ii(p);
	int ip2 = ii(p + group_size - 1);

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
	{
		propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, f, u, k, g);
	}

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

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti)
{

	// number of sensor groups that will be launched in parallel
	int num_groups = NS / group_size;

	// Simulation Variables
	int i = 0, j = 0;

	host_ptr<float>x(NX);
	host_ptr<float>y(NY);
	host_ptr<float>fo(NX, NY);


	dim3 Block_Size(BLOCK_X, BLOCK_Y);
	dim3 Grid_Size(grid_size(NX, BLOCK_X), grid_size(NY, BLOCK_Y));

/*
	dim3 threads_field(NX, 1);
	dim3 grid_field(
		grid_size(NX, threads_field.x),
		grid_size(NY, threads_field.y));
*/
	// Variables of allocation
	device_ptr<float>dev_x(NX);
	device_ptr<float> dev_y (NY);
	device_ptr<float> dev_fo (NX, NY);
	
	device_ptr<float>dev_u (NX, NY, NT, num_groups);

	dev_u.set(0.f);

	device_ptr<float>dev_g_bottom(NX, NT, num_groups);
	device_ptr<float>dev_g_right (NX, NT, num_groups);
	device_ptr<float>dev_g_top(NX, NT, num_groups);
	device_ptr<float> dev_g_left(NX, NT, num_groups);

	dev_g_bottom.set(0.f);
	dev_g_right.set(0.f);
	dev_g_top.set(0.f);
	dev_g_left.set(0.f);

	// Environment Initialization

	for(i = 0; i < NX; i++) {
		x(i) = -0.1f + i * hx;
	}

	for(j = 0; j < NY; j++) {
		y(j) = -0.1f + j * hy;
	}

	copy (dev_x, x);
	copy (dev_y, y);

		field_setup<<<Block_Size, Grid_Size>>>(dev_x, dev_y, dev_fo);
		copy (fo, dev_fo);


	// Position of the transducers
host_ptr<int> ii(NS);
	host_ptr<int> jj(NS);

	Position_Transducers(ii, jj, NS);

	device_ptr<int> dev_ii(NS);
	device_ptr<int> dev_jj(NS);
	copy(dev_ii, ii);
	copy(dev_jj, jj);


	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
		grid_size(NX, threads_propagation.x),
		grid_size(NY, threads_propagation.y));

	dev_u.set(0.f);

	pre_sirt<<<1, num_groups>>>(
		group_size, num_groups,
		dev_ii, dev_jj,
		dev_fo, dev_u,
		dev_g_bottom, dev_g_top,
		dev_g_left, dev_g_right);

	// Kaczmarz method
	// propagation
		
	device_ptr<float> dev_rr_bottom(NX, NT, num_groups);
	device_ptr<float> dev_rr_right(NX, NT, num_groups);
	device_ptr<float> dev_rr_top(NX, NT, num_groups);
	device_ptr<float> dev_rr_left(NX, NT, num_groups);

	dev_rr_bottom.set(0.f);
	dev_rr_right.set(0.f);
	dev_rr_top.set(0.f);
	dev_rr_left.set(0.f);

	device_ptr<float> dev_z(NX, NY, NT+1, num_groups);
	device_ptr<float> dev_Lu(NX, NY, NT, num_groups);
	dev_Lu.set(0.f);

	device_ptr<float> dev_f(NX, NY);
	dev_f.set(0.f);

	device_ptr<float> dev_df(NX, NY);
	device_ptr<float> dev_f_minus_fo(NX, NY);

	// Allocation
	host_ptr<float> f(NX, NY);
	host_ptr<float> f_minus_fo(NX, NY);

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'sirt_convergence.txt'...\n"
		 << "writing time to 'sirt_time.txt'...\n";

	ofstream convergence_file("sirt_convergence.txt");
	ofstream time_file("sirt_time.txt");

	for(int iter = 0; iter < max_iterations; iter++) {
		cout << "\nIter: " << iter << "\n";
		dev_u.set(0.f);
		dev_z.set(0.f);
		dev_df.set(0.f);
		

		sirt<<<1, num_groups>>>
			(group_size, num_groups,
			 dev_ii, dev_jj,
			 dev_f, dev_fo, dev_u,
			 dev_g_bottom, dev_g_top, dev_g_left, dev_g_right,
			 dev_rr_bottom, dev_rr_top, dev_rr_left, dev_rr_right,
			 dev_z, dev_Lu, dev_df, dev_f_minus_fo);

		/* print_differential<<<1, 1>>>(dev_df); */

		update_field<<<Grid_Size, Block_Size>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo, num_groups);

			copy(f_minus_fo, dev_f_minus_fo);

		curr_epsilon = norm<NX, NY>(f_minus_fo) / norm<NX, NY>(fo) * 100.f;
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

	copy(f, dev_f);

	IO_Files<NX, NY>(x, y, fo, f);

	// Free Variables

	size_t free, total;
	cudaMemGetInfo(&free, &total);

	cerr << fixed << setprecision(4);

	cerr << "used mem:  " << float(total - free) / (1024 * 1024) << " MB\n"
		 << "free mem:  " << float(free) / (1024 * 1024)  << " MB\n"
		 << "total mem: " << float(total) / (1024 * 1024) << " MB\n\n";



	cudaDeviceReset();
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

