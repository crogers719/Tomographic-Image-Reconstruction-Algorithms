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
#include "mimo-io.cuh"
using namespace std;

// DEFINITIONS

#define NX 192				//was 201
#define NY 192				//was 201
#define NT 401

#define NS 640 				//number of sensors

#define BLOCK_X 16
#define BLOCK_Y 16

#define HX 0.001f
#define HY 0.001f
#define H 0.001f

/* __constant__ float T = 1.3333e-04f; // 0.2f / 1500.f; */
#define DT 3.3333e-07f
/* __constant__ float fre = 125000.f; */
#define OMEGAC 7.8540e+05f
#define TAO 4.0000e-06f
#define TT 8.1573e-06f

// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, float omega);

float norm(host_ptr<float> A, int nx, int ny)
{
	float sum = 0;

	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i)
			sum += A(i, j) * A(i, j);

	return sqrtf(sum);
}

void Position_Transducers(int *&ii, int *&jj, int num)
{
//returns the (x,y) coordinates of the number of total transducers
	ii = (int*)malloc(num * sizeof(int));
	jj = (int*)malloc(num * sizeof(int));


	for (int p = 0; p < 160; p++)
	{
		ii[p] = 21 + (p + 1);
		jj[p] = 181;
	}

	for (int p = 160; p < 320; p++)
	{
		ii[p] = 181;
		jj[p] = 181 - ((p + 1) - 160);
	}

	for (int p = 320; p < 480; p++)
	{
		ii[p] = 181 - ((p + 1) - 320);
		jj[p] = 21;
	}

	for (int p = 480; p < num; p++)
	{
		ii[p] = 21;
		jj[p] = 21 + ((p + 1) - 480);
	}
}

__global__ void field_setup(kernel_ptr<float> const x, kernel_ptr<float> const y, kernel_ptr<float> fo)
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

__global__ void propagation(
		int jp1, int jp2, int ip1, int ip2, 
		kernel_ptr<float> const f,
		kernel_ptr<float> u, 
		int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i < NX && j < NY) {
		float v = 1500.f * sqrtf(1.f + f(i, j));
		float r = v * DT / HX;
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
				float t = k * DT - TT;

				// add wave value
				val +=
					v * v * DT * DT *
					cosf(OMEGAC * t) *
					expf(-(t * t) / (2.f * TAO * TAO));
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

__global__ void propagation_at_corners(kernel_ptr<float> u)
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
		kernel_ptr<float> const u,
		kernel_ptr<float> g_bottom,
		kernel_ptr<float> g_right,
		kernel_ptr<float> g_top,
		kernel_ptr<float> g_left,
		int p)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	if (i > 20 && i < 180 && k > 1 && k < NT) {
		// store values at bottom sensor row of u
		g_bottom(i, k, p) =
			u(i, 180, k);

		// store values at top sensor row of u
		g_top(i, k, p) =
			u(i, 20, k);

		// store values at right sensor column of u
		g_right(i, k, p) =
			u(180, i, k);

		// store values at left sensor column of u
		g_left(i, k, p) =
			u(20, i, k);
	}
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
		int p)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	if (i > 20 && i < 180 && k > 1 && k < NT) {
		// store difference at time k of original signal
		// and current signal at bottom sensor row
		rr_bottom(i, k) =
			g_bottom(i, k, p) -
			u(i, 180, k);

		/* printf("%e ", rr_bottom(i+21, k+2)); */

		// store difference at time k of original signal
		// and current signal at top sensor row
		rr_top(i, k) =
			g_top(i, k, p) -
			u(i, 20, k);

		// store difference at time k of original signal
		// and current signal at right sensor column
		rr_right(i, k) =
			g_right(i, k, p) -
			u(180, i, k);

		// store difference at time k of original signal
		// and current signal at left sensor column
		rr_left(i, k) =
			g_left(i, k, p) -
			u(20, i, k);
	}
}

__global__ void backpropagation1(
		kernel_ptr<float> z,
		kernel_ptr<float> const f,
		int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if(i >= 1 && i < (NX - 1) && j >= 1 && j < (NY - 1))
	{
		z(i, j, k) =
			1500.f * 1500.f * (DT * DT) *
			((1.f + f(i, j-1)) * z(i, j-1, k+1) +
			 (1.f + f(i, j+1)) * z(i, j+1, k+1) +
			 (1.f + f(i-1, j)) * z(i-1, j, k+1) +
			 (1.f + f(i+1, j)) * z(i+1, j, k+1) -
			 4.f * (1.f + f(i, j)) *
			 z(i, j, k+1)) / (H * H) +
			2.f * z(i, j, k+1) -
			z(i, j, k+2);

		/* if (k == 1) */
			/* printf("%e \t", z(i, j, k)); */
	}
}

__global__ void backpropagation2(
		kernel_ptr<float> z,
		kernel_ptr<float> const rr_bottom,
		kernel_ptr<float> const rr_right,
		kernel_ptr<float> const rr_top,
		kernel_ptr<float> const rr_left,
		int k)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= 21 && i < 180) {
		z(i, 180, k) =
			z(i, 179, k) +
			rr_bottom(i, k) * H * 1000.f;

		z(i, 20, k) =
			z(i, 21, k) +
			rr_top(i, k) * H * 1000.f;

		z(180, i, k) =
			z(179, i, k) +
			rr_right(i, k) * H * 1000.f;

		z(20, i, k) =
			z(21, i, k) +
			rr_left(i, k) * H * 1000.f;
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
		kernel_ptr<float> const u,
		kernel_ptr<float> Lu)
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
			 4.f * u(i, j, k+1)) / (H * H);
	}
}

__global__ void laplace_corners(kernel_ptr<float> const u, kernel_ptr<float> Lu)
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
		kernel_ptr<float> df,
		kernel_ptr<float> const z,
		kernel_ptr<float> const Lu,
		kernel_ptr<float> const f)
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
		kernel_ptr<float> f,
		kernel_ptr<float> const df,
		kernel_ptr<float> f_minus_fo,
		kernel_ptr<float> const fo,
		float omega)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < NX && j < NY)
	{
		bool in_sensor_field = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

		float alpha = in_sensor_field ? 1.f : 0.f;

		f(i, j) += omega * alpha * df(i, j);
		f_minus_fo(i, j) = f(i, j) - fo(i, j);
	}
}


// MAIN PROGRAM

int main(int argc, char **argv)
{
	//Command Line Argument Processing
	if (argc != 5) {
		cerr << "Usage: " << argv[0] << " <sensor group size> <target epsilon> <max iterations> <omega>\n\n";
		exit(1);
	}

	int group_size = stoi(argv[1]);
	float target_epsilon = stof(argv[2]);
	int max_iterations = stoi(argv[3]);
	float omega = stof(argv[4]);

	if (max_iterations == -1)
		max_iterations = numeric_limits<int>::max();

	// set floting-point precision on stdout and stderr
	cout << setprecision(9);
	cerr << setprecision(9);

	Ultrasonic_Tomography(group_size, target_epsilon, max_iterations, omega);
	cudaDeviceReset();
}

inline int grid_size(int n, int threads)
{
	return ceil(float(n) / threads);
}

// FUNCTIONS DEFINITION

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, float omega)
{
	// Simulation Variables

	host_ptr<float> x(NX);
	host_ptr<float> y(NY);
	host_ptr<float> fo(NX, NY);

	dim3 Block_Size(BLOCK_X, BLOCK_Y);
	dim3 Grid_Size(grid_size(NX, BLOCK_X), grid_size(NY, BLOCK_Y));

	device_ptr<float> dev_x(NX);
	device_ptr<float> dev_y(NY);
	device_ptr<float> dev_fo(NX, NY);

	device_ptr<float> dev_u(NX, NY, NT);

	dev_u.set(0.f);

	int Ng = NS / group_size;

	device_ptr<float> dev_g_bottom(NX, NT, Ng);
	device_ptr<float> dev_g_right(NX, NT, Ng);
	device_ptr<float> dev_g_top(NX, NT, Ng);
	device_ptr<float> dev_g_left(NX, NT, Ng);

	dev_g_bottom.set(0.f);
	dev_g_right.set(0.f);
	dev_g_top.set(0.f);
	dev_g_left.set(0.f);

	// Environment Initialization

	for (int i = 0; i < NX; i++)
	{
		x(i) = -0.1f + i * HX;
	}

	for (int j = 0; j < NY; j++)
	{
		y(j) = -0.1f + j * HY;
	}

	copy(dev_x, x);
	copy(dev_y, y);

	field_setup<<<Grid_Size, Block_Size>>>(dev_x, dev_y, dev_fo);

	copy(fo, dev_fo);

	// Position of the transducers
	int *ii, *jj;
	Position_Transducers(ii, jj, NS);

	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
			grid_size(NX, threads_propagation.x),
			grid_size(NY, threads_propagation.y));

	dim3 threads_signal(NX, 1);
	dim3 grid_signal(
			grid_size(NX, threads_signal.x),
			grid_size(NT, threads_signal.y));

	for (int p = 0; p < NS; p += group_size)
	{
		dev_u.set(0.f);

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
			propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, dev_fo, dev_u, k);
		}

		// Four corners

		propagation_at_corners<<<NT, 1>>>(dev_u);

		initial_signal<<<grid_signal, threads_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, p / group_size);
	}

	// Kaczmarz method
	// propagation

	device_ptr<float> dev_rr_bottom(NX, NT);
	device_ptr<float> dev_rr_right(NX, NT);
	device_ptr<float> dev_rr_top(NX, NT);
	device_ptr<float> dev_rr_left(NX, NT);

	dev_rr_bottom.set(0.f);
	dev_rr_right.set(0.f);
	dev_rr_top.set(0.f);
	dev_rr_left.set(0.f);

	device_ptr<float> dev_z(NX, NY, NT+1);
	device_ptr<float> dev_Lu(NX, NY, NT);
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
	float file_epsilon = std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'art_convergence.txt'...\n"
		 << "writing time to 'art_time.txt'...\n\n";

	ofstream convergence_file("art_convergence.txt");
	ofstream time_file("art_time.txt");


	dim3 threads_diff_signal(NX, 1);
	dim3 grid_diff_signal(
			grid_size(NX, threads_diff_signal.x),
			grid_size(NT, threads_diff_signal.y));


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

	cerr << "group size:     " << group_size << "\n"
		 << "target epsilon: " << target_epsilon << "\n"
		 << "omega:          " << omega << "\n\n";


	cout << "iter\tepsilon\n"
		 << "----\t-------\n";

	cudaDeviceSynchronize();
	int ti = clock();

	for (int iter = 0; iter < max_iterations; iter++)
	{
		cout << iter << "\t";
		dev_u.set(0.f);

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

			// Boundary

			for (int k = 1; k < NT - 1; k++)
			{
				propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, dev_f, dev_u, k);
			}

			// Four corners

			propagation_at_corners<<<NT, 1>>>(dev_u);
			difference_signal<<<grid_diff_signal, threads_diff_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, p / group_size);

			dev_z.set(0.f);

			for (int k = NT - 2; k > 0; k--)
			{
				backpropagation1<<<grid_backpropagation1, threads_backpropagation1>>>(dev_z, dev_f, k);
				backpropagation2<<<NX, 1>>>(dev_z, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, k);
			}

			laplace<<<grid_laplace, threads_laplace>>>(dev_u, dev_Lu);
			laplace_corners<<<NT, 1>>>(dev_u, dev_Lu);

			dev_df.set(0.f);
			update_differential<<<grid_differential, threads_differential>>>(dev_df, dev_z, dev_Lu, dev_f);

			update_field<<<Grid_Size, Block_Size>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo, omega);
		}

		copy(f_minus_fo, dev_f_minus_fo);

		curr_epsilon = norm(f_minus_fo, NX, NY) / norm(fo, NX, NY) * 100.f;
		float current_t = (float)(clock()-ti) / CLOCKS_PER_SEC;

		if (file_epsilon - curr_epsilon > 0.2f) {
			convergence_file << curr_epsilon << " ";
			time_file << current_t << " ";
			file_epsilon = curr_epsilon;
		}

		cout << curr_epsilon << "\n";

		// stop if reached target epsilon
		if (curr_epsilon <= target_epsilon) {
			cerr << "reached target epsilon = " << target_epsilon << ", at iter = " << iter << ", epsilon = " << curr_epsilon << "\n\n";
			break;
		}

		// stop if epsilon diverges
		if (curr_epsilon > prev_epsilon ||
				std::isnan(curr_epsilon)) {
			cerr << "diverged at iter = " << iter << ", epsilon = " << curr_epsilon << "\n\n";
			break;
		}

		// update prev_epsilon
		prev_epsilon = curr_epsilon;
	}

	cudaDeviceSynchronize();
	int tf = clock();

	cout << endl;

	// copy from device to host
	copy(f, dev_f);

	string f_name = "art-" + to_string(group_size) + "-f.txt";
	cerr << "writing to '" << f_name << "'...\n\n";

	ofstream f_out(f_name);
	write(f_out, f);

	// Free Variables

	delete [] ii;
	delete [] jj;

	cerr << "time (s): " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";
}
