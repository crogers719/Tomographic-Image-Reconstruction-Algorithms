/**********HEADERS**********/

#include <algorithm>
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
void Ultrasonic_Tomography(const string&, int, int, float);
void Position_Transducers(host_ptr<int>, host_ptr<int>, int);

//In-Line Functions
inline int grid_size(int, int);
template <typename T> __host__ __device__ void minmax(T &a, T &b);

//Device Functions
__global__ void propagation(kernel_ptr<int> const, kernel_ptr<int> const,	kernel_ptr<float> const, kernel_ptr<float>,	int, int, int);
__global__ void propagation_at_corners(kernel_ptr<float>,	int);
__global__ void initial_signal(kernel_ptr<float> const,	kernel_ptr<float>, kernel_ptr<float>,	kernel_ptr<float>,kernel_ptr<float>,int);


/***************MAIN PROGRAM***************/

int main(int argc, char **argv)
{
	//Command Line Argument Processing
	if (argc != 4) {
		cerr << "Usage: " << argv[0] << " <fo filename> <sensor group size> <percent noise>\n\n";
		exit(1);
	}

	string fo_filename = argv[1];
	int group_size = stoi(argv[2]);
	float percent_noise = (stoi (argv[3]))/100.00f;

	printf ("Percent Noise = %.2f\n\n", percent_noise);

	if (count(fo_filename.begin(), fo_filename.end(), '.') != 1) {
		cerr << "Error: '" << fo_filename << "' should have only one period.\n"
			<< "       It should be in the current directory "
			<< "and have only one filetype extension.\n\n";
		exit(1);
	}

	// Time Measuring Variables
	int ti = 0, tf = 0;

	// set floating-point precision on stdout and stderr
	cout << fixed << setprecision(10);
	cerr << fixed << setprecision(10);

	cerr << "Ultrasonic Tomography Running:\n\n";

	//Initial time
	ti = clock();
	cerr << "ti = " << ti << "\n";

	Ultrasonic_Tomography(fo_filename, group_size, ti, percent_noise);
	cudaDeviceReset();

	//Calculate total time
	tf = clock();
	cerr << "tf = " << tf << "\n"
		 << "tt = " << tf - ti << "\n"
		 << "Total Seconds = " << (float)(tf - ti) / CLOCKS_PER_SEC << "\n";
}



/**********HOST FUNCTION DEFINITIONS**********/

void Ultrasonic_Tomography(const string &fo_filename, int group_size, int ti, float percent_noise)
{
	// environment initialization

	// fo(i, j) =
	//    ground truth value at pos (i, j) of field
	host_ptr<float> fo(NX, NY);
	device_ptr<float> dev_fo(NX, NY);

	{
		ifstream fo_in(fo_filename);

		if (!fo_in) {
			cerr << "Error: '" + fo_filename + "' file not found in current directory.\n\n";
			return;
		}

		read(fo_in, fo);
		copy(dev_fo, fo);
	}

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
	
	host_ptr<float>u(NX, NY, NT, Ng);
	copy (u, dev_u)	;

	
	for (int x =0; x< NX; x++)
	{
		for (int y=0; y<NY; y++)
		{
			for (int t=0; t<NT; t++)
			{
				for (int g=0; g<Ng; g++)
				{
					float noise = (1.0f- percent_noise) + (percent_noise *2)*((float)rand()/RAND_MAX);
					
					//if (u(x,y,t,g)!=0){
					//	printf("Before: %.10f\t\t\t\t || \t\t\t\t", u(x,y,t,g));					
						u(x, y, t, g)=u(x, y, t, g) *noise;
						//printf ("After : %.10f\t\t\t\tNoise:%.2f\n\n", u(x,y,t,g), noise);
					//} 
				}
			}
		}	
	}
	copy(dev_u, u);

	// store initial signal of wave at sensor positions of u in g
	initial_signal<<<grid_signal, threads_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, Ng);
	

	host_ptr<float> g_bottom(NX, NT, Ng);
	host_ptr<float> g_right(NY, NT, Ng);
	host_ptr<float> g_top(NX, NT, Ng);
	host_ptr<float> g_left(NY, NT, Ng);
	
	copy(g_bottom, dev_g_bottom);
	copy(g_right, dev_g_right);
	copy(g_top, dev_g_top);
	copy(g_left, dev_g_left);
	/*
	for (int x =0; x< NX; x++)
	{
		for (int t=0; t<NT; t++)
		{
			for (int g=0; g<Ng; g++)
			{
				float noise = (1- percent_noise) + (percent_noise *2)*((float)rand()/RAND_MAX);
				g_bottom(x, t, g) = g_bottom(x, t, g) * noise;
				g_right (x,t,g)=g_right(x,t,g)*noise;
				g_top(x,t,g)=g_top (x,t,g)*noise;
				g_left(x,t,g)=g_left(x,t,g)*noise;
			}
		}	
	}

	copy(dev_g_bottom, g_bottom);
	copy(dev_g_right, g_right);
	copy(dev_g_top, g_top);
	copy(dev_g_left, g_left);

*/


	{
		auto idx = fo_filename.find_first_of('.');
		string prefix = fo_filename.substr(0, idx) + "-data-";
		string suffix = "-" + to_string(group_size) + ".txt";

		string gb_name = prefix + "bottom" + suffix;
		string gr_name = prefix + "right" + suffix;
		string gt_name = prefix + "top" + suffix;
		string gl_name = prefix + "left" + suffix;

		ofstream gb_out(gb_name);
		ofstream gr_out(gr_name);
		ofstream gt_out(gt_name);
		ofstream gl_out(gl_name);

		cerr << "writing to '" << gb_name << "'...\n\n";
		write(gb_out, g_bottom);

		cerr << "writing to '" << gr_name << "'...\n\n";
		write(gr_out, g_right);

		cerr << "writing to '" << gt_name << "'...\n\n";
		write(gt_out, g_top);

		cerr << "writing to '" << gl_name << "'...\n\n";
		write(gl_out, g_left);
	}
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

		float val; // will hold new u at (i, j, k + 1, g)

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


