// HEADERS

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>
#include <random>

#include "cuda_ptr.cuh"
#include "mimo-io.cuh"
#include "generator.cuh"
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

#define DT 3.3333e-07f
#define OMEGAC 7.8540e+05f
#define TAO 4.0000e-06f
#define TT 8.1573e-06f

// FUNCTIONS DECLARATION

void Ultrasonic_Tomography(const string&, int, int , float);

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


// MAIN PROGRAM

int main(int argc, char **argv)
{
	//Command Line Argument Processing
	if (argc != 4) {
		cerr << "Usage: " << argv[0] << " <fo_filename> <sensor group size> <percent noise> \n\n";
		exit(1);
	}

	string fo_filename = argv[1];
	int group_size = stoi(argv[2]);
	float percent_noise = (stoi (argv[3]))/100.00f;

	if (count(fo_filename.begin(), fo_filename.end(), '.') != 1) {
		cerr << "Error: '" << fo_filename << "' should have only one period.\n"
			<< "       It should be in the current directory "
			<< "and have only one filetype extension.\n\n";
		exit(1);
	}

	cout << setprecision(9);
	cerr << setprecision(9);

	// Time measuring variables

	int ti = 0, tf = 0;

	// Function Execution

	ti = clock();
	cerr << "ti = " << ti << "\n";

	Ultrasonic_Tomography(fo_filename, group_size, ti, percent_noise);
	cudaDeviceReset();

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

	Position_Transducers(ii, jj, NS);

	device_ptr<float> dev_u(NX, NY, NT);

	int Ng = NS / group_size;

	device_ptr<float> dev_g_bottom(NX, NT, Ng);
	device_ptr<float> dev_g_right(NY, NT, Ng);
	device_ptr<float> dev_g_top(NX, NT, Ng);
	device_ptr<float> dev_g_left(NY, NT, Ng);

	host_ptr<float> g_bottom(NX, NT, Ng);
	host_ptr<float> g_right(NY, NT, Ng);
	host_ptr<float> g_top(NX, NT, Ng);
	host_ptr<float> g_left(NY, NT, Ng);

	dev_g_bottom.set(0.f);
	dev_g_right.set(0.f);
	dev_g_top.set(0.f);
	dev_g_left.set(0.f);

	// kernel launch parameters for propagation
	dim3 threads_propagation(NX, 1);
	dim3 grid_propagation(
			grid_size(NX, threads_propagation.x),
			grid_size(NY, threads_propagation.y));

	// kernel launch parameters for propagation_at_corners
	dim3 threads_prop_corners(NT, 1);
	dim3 grid_prop_corners(
			grid_size(NT, threads_prop_corners.x));

	dim3 threads_signal(NX, 1);
	dim3 grid_signal(
			grid_size(NX, threads_signal.x),
			grid_size(NT, threads_signal.y));

	Generator gen (-1*percent_noise, percent_noise);


	for (int p = 0; p < NS; p += group_size)
	{
		dev_u.set(0.f);

		int group = p/group_size;

		int jp1 = jj(p);
		int jp2 = jj(p + group_size - 1);
		int ip1 = ii(p);
		int ip2 = ii(p + group_size - 1);

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

		for (int k = 1; k < NT - 1; k++)
		{
			propagation<<<grid_propagation, threads_propagation>>>(jp1, jp2, ip1, ip2, dev_fo, dev_u, k);
		}

		propagation_at_corners<<<grid_prop_corners, threads_prop_corners>>>(dev_u);

		initial_signal<<<grid_signal, threads_signal>>>(dev_u, dev_g_bottom, dev_g_right, dev_g_top, dev_g_left, group);

	copy(g_bottom, dev_g_bottom);
	copy(g_right, dev_g_right);
	copy(g_top, dev_g_top);
	copy(g_left, dev_g_left);
	
//ADDING NOISE
/*
	
 	float p_noise; 
	float py =0.0;

	for (int x =0; x< NX; x++)
	{
		for (int t=0; t<NT; t++)
		{
				py += pow(g_bottom (x,t,group), 2);
				py += pow(g_right(x,t,group), 2);
				py += pow(g_top (x,t,group), 2);
				py += pow(g_left (x,t,group), 2);
		}
	}

	p_noise= percent_noise * py;
	float stddev= sqrt(p_noise); 

	printf("\nPercent Noise= %.2f \nP Noise=  %.100f \npy=%.100f \nStddev=%.10f\n\n", percent_noise, p_noise, py, stddev);

	default_random_engine generator;  
	normal_distribution <float> distribution (0, stddev);
*/



	for (int x=0; x<NX; x++)
	{ 
		for (int t=0; t<NT; t++)
		{
				g_bottom (x,t,group) += g_bottom(x, t, group) * gen();
				g_right (x,t,group) += g_right(x, t, group) * gen();
				g_top (x,t,group) += g_top(x, t, group) * gen();
				g_left (x, t, group) += g_left (x, t, group) * gen();
		}
	}

	copy(dev_g_bottom, g_bottom);
	copy(dev_g_right, g_right);
	copy(dev_g_top, g_top);
	copy(dev_g_left, g_left);

}


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
