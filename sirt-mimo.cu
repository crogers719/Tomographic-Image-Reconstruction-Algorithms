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
using namespace std;

/**********DEFINING CONSTANTS***********/

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

/**********FUNCTION DECLARATION**********/

//Host Functions
void Ultrasonic_Tomography(int, float, int, int, float);
void Position_Transducers(host_ptr<int>, host_ptr<int>, int);
void IO_Files(host_ptr<float> const, host_ptr<float> const);
float norm(host_ptr<float>, int, int);

//In-Line Functions
inline int grid_size(int, int);
template <typename T> __host__ __device__ void minmax(T &a, T &b);

//Device Functions
__global__ void field_setup(kernel_ptr<float> const, kernel_ptr<float> const, kernel_ptr<float>);
__global__ void propagation(kernel_ptr<int> const, kernel_ptr<int> const,	kernel_ptr<float> const, kernel_ptr<float>,	int, int, int);
__global__ void propagation_at_corners(kernel_ptr<float>,	int);
__global__ void initial_signal(kernel_ptr<float> const,	kernel_ptr<float>, kernel_ptr<float>,	kernel_ptr<float>,kernel_ptr<float>,int);
__global__ void difference_signal(kernel_ptr<float> const,kernel_ptr<float> const, kernel_ptr<float> const, kernel_ptr<float> const, kernel_ptr<float> const,	kernel_ptr<float>,kernel_ptr<float>,	kernel_ptr<float>, kernel_ptr<float>,	int);
__global__ void backpropagation1(kernel_ptr<float>, kernel_ptr<float> const,int, int);
__global__ void backpropagation2(kernel_ptr<float>,	kernel_ptr<float> const, kernel_ptr<float> const,	kernel_ptr<float> const,kernel_ptr<float> const, int, int);
__global__ void laplace(kernel_ptr<float> cons,	kernel_ptr<float>, int);
__global__ void laplace_corners(kernel_ptr<float> const, kernel_ptr<float>,	int);
__global__ void update_differential(kernel_ptr<float>,kernel_ptr<float>,kernel_ptr<float> const, kernel_ptr<float> const ,int);
__global__ void update_field(kernel_ptr<float> f, kernel_ptr<float> const df, kernel_ptr<float> f_minus_fo, kernel_ptr<float> const fo,	float omega);



/***************MAIN PROGRAM***************/

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

	// Time Measuring Variables
	int ti = 0, tf = 0;

	// set floating-point precision on stdout and stderr
	cout << fixed << setprecision(10);
	cerr << fixed << setprecision(10);

	cerr << "Ultrasonic Tomography Running:\n\n";

	//Initial time
	ti = clock();
	cerr << "ti = " << ti << "\n";

	Ultrasonic_Tomography(group_size, target_epsilon, max_iterations, ti, omega);
	cudaDeviceReset();

	//Calculate total time
	tf = clock();
	cerr << "tf = " << tf << "\n"
		 << "tt = " << tf - ti << "\n"
		 << "Total Seconds = " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";
}



/**********HOST FUNCTION DEFINITIONS**********/

void Ultrasonic_Tomography(int group_size, float target_epsilon, int max_iterations, int ti, float omega)
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

	// Position of the transducers
	host_ptr<int> ii(NS);
	host_ptr<int> jj(NS);
	device_ptr<int> dev_ii(NS);
	device_ptr<int> dev_jj(NS);

	Position_Transducers(ii, jj, NS);

	// copy from host to device
	copy(dev_ii, ii);
	copy(dev_jj, jj);

	// Ng = number of sensor groups that will be launched in parallel
	int Ng = NS / group_size;

	// u(i, j, k, g) =
	//    wave propagation at pos (i, j) of field, at time k, from sensor group g
	device_ptr<float> dev_u(NX, NY, NT, Ng);
	dev_u.set(0.f);

	// kernel launch parameters for propagation
	dim3 threads_propagation(NX, 1, 1);
	dim3 grid_propagation(
			grid_size(NX, threads_propagation.x),
			grid_size(NY, threads_propagation.y),
			grid_size(Ng, threads_propagation.z));

	// kernel launch parameters for propagation_at_corners
	dim3 threads_prop_corners(NT, 1);
	dim3 grid_prop_corners(
			grid_size(NT, threads_prop_corners.x),
			grid_size(Ng, threads_prop_corners.y));

	// initial wave propagation over fo
	for (int k = 1; k < NT - 1; ++k)
		propagation<<<grid_propagation, threads_propagation>>>(dev_ii, dev_jj, dev_fo, dev_u, k, group_size, Ng);

	propagation_at_corners<<<grid_prop_corners, threads_prop_corners>>>(dev_u, Ng);

	// gg_xxx(i, k, g) =
	//    initial signal at pos i in row/column xxx
	//    at time k, from sensor group
	//    e.g g_bottom stores the bottom row,
	//        g_right stores the right column
	device_ptr<float> dev_g_bottom(NX, NT, Ng);
	device_ptr<float> dev_g_right(NY, NT, Ng);
	device_ptr<float> dev_g_top(NX, NT, Ng);
	device_ptr<float> dev_g_left(NY, NT, Ng);

	dev_g_bottom.set(0.f);
	dev_g_right.set(0.f);
	dev_g_top.set(0.f);
	dev_g_left.set(0.f);

	// kernel launch parameters for initial_signal
	dim3 threads_signal(NX, 1, 1);
	dim3 grid_signal(
			grid_size(NX, threads_signal.x),
			grid_size(NT, threads_signal.y),
			grid_size(Ng, threads_signal.z));

	// store initial signal of wave at sensor positions of u in g
	initial_signal<<<grid_signal, threads_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, Ng);

	// Kaczmarz method
	// propagation

	// rr_xxx(i, k, g) =
	//    difference signal between xxx sensors in u and gg_xxx
	//    at time k, from sensor group g
	device_ptr<float> dev_rr_bottom(NX, NT, Ng);
	device_ptr<float> dev_rr_right(NX, NT, Ng);
	device_ptr<float> dev_rr_top(NX, NT, Ng);
	device_ptr<float> dev_rr_left(NX, NT, Ng);

	dev_rr_bottom.set(0.f);
	dev_rr_right.set(0.f);
	dev_rr_top.set(0.f);
	dev_rr_left.set(0.f);

	// z(i, j, k, g) =
	//    wave back propagation at pos (i, j) of field,
	//    at time k, from sensor group g
	device_ptr<float> dev_z(NX, NY, NT+1, Ng);

	// Lu(i, j, k, g) =
	//    result of applying the Laplace operator to u(i, j, k, g)
	device_ptr<float> dev_Lu(NX, NY, NT, Ng);
	dev_Lu.set(0.f);

	// f(i, j) =
	//    current reconstruction of field at pos (i, j)
	host_ptr<float> f(NX, NY);
	device_ptr<float> dev_f(NX, NY);
	dev_f.set(0.f);

	// df(i, j) =
	//    discretized differential of f(i, j)
	device_ptr<float> dev_df(NX, NY);

	// f_minus_fo(i, j)
	//    difference of field and ground truth at pos (i, j)
	host_ptr<float> f_minus_fo(NX, NY);
	device_ptr<float> dev_f_minus_fo(NX, NY);

	// initialize epsilon values
	float prev_epsilon = std::numeric_limits<float>::infinity();
	float curr_epsilon = -std::numeric_limits<float>::infinity();
	float file_epsilon = std::numeric_limits<float>::infinity();

	cerr << "writing convergence to 'sirt_convergence.txt'...\n"
		 << "writing time to 'sirt_time.txt'...\n";

	ofstream convergence_file("sirt_convergence.txt");
	ofstream time_file("sirt_time.txt");

	// kernel launch parameters for difference_signal
	dim3 threads_diff_signal(NX, 1, 1);
	dim3 grid_diff_signal(
			grid_size(NX, threads_diff_signal.x),
			grid_size(NT, threads_diff_signal.y),
			grid_size(Ng, threads_diff_signal.z));

	// kernel launch parameters for backpropagation1
	dim3 threads_backpropagation1(NX, 1, 1);
	dim3 grid_backpropagation1(
			grid_size(NX, threads_backpropagation1.x),
			grid_size(NY, threads_backpropagation1.y),
			grid_size(Ng, threads_backpropagation1.z));

	// kernel launch parameters for backpropagation2
	dim3 threads_backpropagation2(Ng, 1);
	dim3 grid_backpropagation2(
			grid_size(NX, threads_backpropagation2.x),
			grid_size(Ng, threads_backpropagation2.y));

	// kernel launch parameters for laplace
	dim3 threads_laplace(NX, 1, 1);
	dim3 grid_laplace(
			grid_size(NX * NY, threads_laplace.x),
			grid_size(NT, threads_laplace.y),
			grid_size(Ng, threads_laplace.z));

	// kernel launch parameters for laplace_corners
	dim3 threads_laplace_corners(NX, 1, 1);
	dim3 grid_laplace_corners(
			grid_size(NX * NY, threads_laplace.x),
			grid_size(NT, threads_laplace.y),
			grid_size(Ng, threads_laplace.z));

	// kernel launch parameters for update_differential
	dim3 threads_differential(NX, 1, 1);
	dim3 grid_differential(
			grid_size(NX * NY, threads_differential.x),
			grid_size(NT, threads_differential.y),
			grid_size(Ng, threads_differential.z));

	cerr << "group size:     " << group_size << "\n"
		 << "target epsilon: " << setprecision(2) << target_epsilon << "\n"
		 << "omega:          " << setprecision(0) << omega << "\n\n";

	for(int iter = 0; iter < max_iterations; iter++)
	{
		cout << "\nIter: " << iter << "\n";
	
		dev_u.set(0.f);
		dev_z.set(0.f);
		dev_df.set(0.f);

		// propagate wave over field, store in u
		for (int k = 1; k < NT - 1; ++k)
			propagation<<<grid_propagation, threads_propagation>>>(dev_ii, dev_jj, dev_f, dev_u, k, group_size, Ng);

		propagation_at_corners<<<grid_prop_corners, threads_prop_corners>>>(dev_u, Ng);

		// store difference signal of u at sensor positions and initial signal at g in rr
		difference_signal<<<grid_diff_signal, threads_diff_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, Ng);

		// do back propagation of wave over field, store in z
		for(int k = NT - 2; k > 0; k--)
		{
			backpropagation1<<<grid_backpropagation1, threads_backpropagation1>>>(dev_z, dev_f, k, Ng);
			backpropagation2<<<grid_backpropagation2, threads_backpropagation2>>>(dev_z, dev_rr_bottom, dev_rr_right, dev_rr_top, dev_rr_left, k, Ng);
		}

		// apply Laplace operator to u, store in Lu
		laplace<<<grid_laplace, threads_laplace>>>(dev_u, dev_Lu, Ng);
		laplace_corners<<<grid_laplace_corners, threads_laplace_corners>>>(dev_u, dev_Lu, Ng);

		// update differential of f, store in df
		update_differential<<<grid_differential, threads_differential>>>(dev_df, dev_z, dev_Lu, dev_f, Ng);

		// update f and f_minus_fo
		update_field<<<grid_field, threads_field>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo, omega);

		// error calculation

		// copy from device to host
		copy(f_minus_fo, dev_f_minus_fo);

		curr_epsilon = norm(f_minus_fo, NX, NY) / norm(fo, NX, NY) * 100.f;
		float current_t = (float)(clock()-ti) / CLOCKS_PER_SEC;

		if (file_epsilon - curr_epsilon > 0.2f) {
			convergence_file << curr_epsilon << " ";
			time_file << current_t << " ";
			file_epsilon = curr_epsilon;
		}

		cout << "epsilon = " << setprecision(8) << curr_epsilon << "\n";

		// stop if reached target epsilon
		if (curr_epsilon <= target_epsilon) {
			cerr << "reached target epsilon = " << setprecision(2) << target_epsilon << ", at iter = " << iter << ", epsilon = " << setprecision(8) << curr_epsilon << "\n\n";
			break;
		}

		// stop if epsilon diverges
		if (curr_epsilon > prev_epsilon ||
				std::isnan(curr_epsilon)) {
			cerr << "diverged at iter = " << iter << ", epsilon = " << setprecision(8) << curr_epsilon << "\n\n";
			break;
		}

		// update prev_epsilon
		prev_epsilon = curr_epsilon;
	}

	cout << endl;

	// copy from device to host
	copy(f, dev_f);

	IO_Files(fo, f);

	size_t free, total;
	cudaMemGetInfo(&free, &total);

	cerr << fixed << setprecision(4);

	cerr << "used mem:  " << float(total - free) / (1024 * 1024) << " MB\n"
		 << "free mem:  " << float(free) / (1024 * 1024)  << " MB\n"
		 << "total mem: " << float(total) / (1024 * 1024) << " MB\n\n";
}

void IO_Files(host_ptr<float> const fo, host_ptr<float> const f)
{
	int i = 0, j = 0;

	// IO Files
	ofstream fo_file;
	ofstream f_file;

	cerr << "writing f0 to 'dev_f0.txt'...\n"
		 << "writing f to 'dev_f.txt'...\n\n";

	//ground truth file
	fo_file.open("dev_f0.txt");
	//reconstructed image file
	f_file.open("dev_f.txt");

	for(j = 0; j < NY; j++) {
		for(i = 0; i < NX; i++) {
			fo_file << fo(i, j);
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(j = 0; j < NY; j++) {
		for(i = 0; i < NX; i++) {
			f_file << f(i, j);
			f_file << " ";
		}

		f_file << "\n";
	}

	fo_file.close();
	f_file.close();
}

float norm(host_ptr<float> A, int nx, int ny)
{
	float sum = 0;

	for (int j = 0; j < ny; ++j)
		for (int i = 0; i < nx; ++i)
			sum += A(i, j) * A(i, j);

	return sqrtf(sum);
}

void Position_Transducers(host_ptr<int> ii, host_ptr<int> jj, int num)
{
	//returns the (x,y) coordinates of the number of total transducers
	int p = 0;

	for(p = 0; p < 160; p++)
	{
		ii(p) = 21 + (p + 1);
		jj(p) = 181;
	}

	for(p = 160; p < 320; p++)
	{
		ii(p) = 181;
		jj(p) = 181 - ((p + 1) - 160);
	}

	for(p = 320; p < 480; p++)
	{
		ii(p) = 181 - ((p + 1) - 320);
		jj(p) = 21;
	}

	for(p = 480; p < num; p++)
	{
		ii(p) = 21;
		jj(p) = 21 + ((p + 1) - 480);
	}
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
	kernel_ptr<int> const ii,
	kernel_ptr<int> const jj,
	kernel_ptr<float> const f,
	kernel_ptr<float> u,
	int k, int group_size, int Ng)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if(i < NX && j < NY && g < Ng) {
		float v = 1500.f * sqrtf(1.f + f(i, j));
		float r = v * DT / HX;
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

			int p = g * group_size;

			int jp1 = jj(p);
			int jp2 = jj(p + group_size - 1);
			int ip1 = ii(p);
			int ip2 = ii(p + group_size - 1);

			minmax(jp1, jp2);
			minmax(ip1, ip2);

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
				(2.f - 2.f * r - r * r) * u(i, j, k, g) +
				2.f * r * (1.f + r) * u(ia, ja, k, g) -
				r * r * u(ib, jb, k, g) +
				(2.f * r - 1.f) * u(i, j, k-1, g) -
				2.f * r * u(ia, ja, k-1, g);
		}

		u(i, j, k+1, g) = val;


	}
}

__global__ void propagation_at_corners(
	kernel_ptr<float> u,
	int Ng)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int g = threadIdx.y + blockIdx.y * blockDim.y;

	if (k < NT && g < Ng) {
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
	int Ng)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if (i > 20 && i < 180 && k > 1 && k < NT && g < Ng) {
		// store values at bottom sensor row of u
		g_bottom(i, k, g) =
			u(i, 180, k, g);

		// store values at top sensor row of u
		g_top(i, k, g) =
			u(i, 20, k, g);



		// store values at right sensor column of u
		g_right(i, k, g) =
			u(180, i, k, g);


		// store values at left sensor column of u
		g_left(i, k, g) =
			u(20, i, k, g);
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
	int Ng)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if (i > 20 && i < 180 && k > 1 && k < NT && g < Ng) {
		// store difference at time k of original signal
		// and current signal at bottom sensor row
		rr_bottom(i, k, g) = g_bottom(i, k, g) - u(i, 180, k, g);

		// store difference at time k of original signal
		// and current signal at top sensor row
		rr_top(i, k, g) = g_top(i, k, g) - u(i, 20, k, g);

		// store difference at time k of original signal
		// and current signal at right sensor column
		rr_right(i, k, g) = g_right(i, k, g) - u(180, i, k, g);

		// store difference at time k of original signal
		// and current signal at left sensor column
		rr_left(i, k, g) = g_left(i, k, g) - u(20, i, k, g);
	}
}

__global__ void backpropagation1(
	kernel_ptr<float> z,
	kernel_ptr<float> const f,
	int k, int Ng)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if(i >= 1 && i < (NX - 1) && j >= 1 && j < (NY - 1) && g < Ng)
	{
		z(i, j, k, g) =
			1500.f * 1500.f * (DT * DT) *
			((1.f + f(i, j-1)) * z(i, j-1, k+1, g) +
			 (1.f + f(i, j+1)) * z(i, j+1, k+1, g) +
			 (1.f + f(i-1, j)) * z(i-1, j, k+1, g) +
			 (1.f + f(i+1, j)) * z(i+1, j, k+1, g) -
			 4.f * (1.f + f(i, j)) *
			 z(i, j, k+1, g)) / (H * H) +
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
	int k, int Ng)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int g = threadIdx.y + blockIdx.y * blockDim.y;

	if (g < Ng) {
		if(i >= 21 && i < 180) {
			z(i, 180, k, g) =
				z(i, 179, k, g) +
				rr_bottom(i, k, g) * H * 1000.f;

			z(i, 20, k, g) =
				z(i, 21, k, g) +
				rr_top(i, k, g) * H * 1000.f;

			z(180, i, k, g) =
				z(179, i, k, g) +
				rr_right(i, k, g) * H * 1000.f;

			z(20, i, k, g) =
				z(21, i, k, g) +
				rr_left(i, k, g) * H * 1000.f;
		}

		if (i >= 1 && i < (NX - 1)) {
			z(i, 0, k, g) =
				z(i, 1, k, g);

			z(i, NY-1, k, g) =
				z(i, NY-2, k, g);

			z(0, i, k, g) =
				z(1, i, k, g);

			z(NX-1, i, k, g) =
				z(NX-2, i, k, g);
		}

		else if (i == 0) {
			z(0, 0, k, g) =
				(z(1, 0, k, g) +
				 z(0, 1, k, g)) / 2.f;

			z(NX-1, 0, k, g) =
				(z(NX-2, 0, k, g) +
				 z(NX-1, 1, k, g)) / 2.f;

			z(0, NY-1, k, g) =
				(z(1, NY-1, k, g) +
				 z(0, NY-2, k, g)) / 2.f;

			z(NX-1, NY-1, k, g) =
				(z(NX-2, NY-1, k, g) +
				 z(NX-1, NY-2, k, g)) / 2.f;
		}
	}
}

__global__ void laplace(
	kernel_ptr<float> const u,
	kernel_ptr<float> Lu,
	int Ng)
{
	// Map from threadIdx / BlockIdx to pixel position

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if (tx < (NX * NY) && (k + 1) < NT && g < Ng) {
		int i = tx % NX;
		int j = tx / NX;

		int ja = (j > 0) ? (j - 1) : j;
		int jb = (j < NY - 1) ? (j + 1) : j;

		int ia = (i > 0) ? (i - 1) : i;
		int ib = (i < NX - 1) ? (i + 1) : i;

		Lu(i, j, k+1, g) =
			(u(i, ja, k+1, g) +
			 u(i, jb, k+1, g) +
			 u(ia, j, k+1, g) +
			 u(ib, j, k+1, g) -
			 4.f * u(i, j, k+1, g)) / (H * H);
	}
}

__global__ void laplace_corners(
	kernel_ptr<float> const u,
	kernel_ptr<float> Lu,
	int Ng)
{
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int g = threadIdx.y + blockIdx.y * blockDim.y;

	if ((k + 1) < NT && g < Ng) {
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
	int Ng)
{
	// Map from threadIdx / BlockIdx to pixel position

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int g = threadIdx.z + blockIdx.z * blockDim.z;

	if (tx < (NX * NY) && (k + 1) < NT && g < Ng) {
		int i = tx % NX;
		int j = tx / NX;

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

/**********INLINE FUNCTION DEFINITIONS**********/
inline int grid_size(int n, int threads)
{
	return ceil(float(n) / threads);
}


// POST-CONDITION: a <= b
template <typename T>
__host__ __device__ 
void minmax(T &a, T &b)
{
	if (a > b) {
		int t = a;
		a = b;
		b = t;
	}
}


