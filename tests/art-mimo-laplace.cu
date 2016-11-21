// HEADERS

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;

// DEFINITIONS

#define NX		201
#define NY		201
#define NT		401
#define SENSOR_GROUP_SIZE		10

#ifndef I_MAX
#define I_MAX 20
#endif /* I_MAX */

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

void Ultrasonic_Tomography();
void IO_Files(float*, float*, float*, float*);
float norm(float*, int);

__global__ void field_setup(const float*, const float*, float*);
__global__ void propagation(int, int, int, int, const float*, float*, int);
__global__ void propagation_at_corners(float*);
__global__ void initial_signal(const float*, float*, float*, float*, float*, int);
__global__ void difference_signal(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, float*, int);
__global__ void backpropagation1(float*, const float*, int);
__global__ void backpropagation2(float*, const float*, const float*, const float*, const float*, int);
__global__ void laplace1(const float*, float*);
__global__ void laplace2(const float*, float*);
__global__ void init_differential(float*, const float*, const float*, const float*);
__global__ void update_differential(float*, const float*, const float*, const float*, int);
/* __global__ void update_field(float*, float*, const float*, float*, const float*); */
__global__ void update_field(float*, const float*, float*, const float*);
__global__ void reset(const float*, float*, float*, float*, float*);

// MAIN PROGRAM

int main(void)
{
	// Time measuring variables

	int ti = 0, tf = 0;

	// Function Execution

	printf("Ultrasonic Tomography Running:\n\n");

	ti = clock();
	printf("ti = %d\n", ti);

	Ultrasonic_Tomography();

	tf = clock();
	printf("tf = %d\n", tf);
	printf("tt = %d\n", tf - ti);
	printf("Total Seconds = %f\n", (float)(tf - ti)  / CLOCKS_PER_SEC);

	cudaDeviceReset();

	// End of the program

	/* system("pause"); */
	return 0;
}

// FUNCTIONS DEFINITION

void Ultrasonic_Tomography()
{
	// Simulation Variables

	float hx = 0.001f;
	float hy = 0.001f;

	int i = 0, j = 0, k = 0;
	int Nx_Ny = NX * NY;
	int Nx_Ny_Nt = NX * NY * NT;
	int Nx_Nt = NX * NT;

	float *x = new float[NX];
	float *y = new float[NY];
	float *fo = new float[Nx_Ny];
	float *u = new float[Nx_Ny_Nt];

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

	float *dev_x;
	int size_x = NX * sizeof(float);

	float *dev_y;
	int size_y = NX * sizeof(float);

	float *dev_fo;
	int size_fo = Nx_Ny * sizeof(float);

	float *dev_u;
	int size_u = Nx_Ny_Nt * sizeof(float);

	float *dev_g1;
	int size_g1 = Nx_Nt * 640 * sizeof(float);

	float *dev_g2;
	int size_g2 = Nx_Nt * 640 * sizeof(float);

	float *dev_g3;
	int size_g3 = Nx_Nt * 640 * sizeof(float);

	float *dev_g4;
	int size_g4 = Nx_Nt * 640 * sizeof(float);

	cudaMalloc((void**) &dev_x, size_x);
	cudaMalloc((void**) &dev_y, size_y);
	cudaMalloc((void**) &dev_fo, size_fo);
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

	int p = 0;
	int *jj = new int[640];
	int *ii = new int[640];

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

	for(p = 480; p < 640; p++)
	{
		ii[p] = 21;
		jj[p] = 21 + ((p + 1) - 480);
	}

	for(p = 0; p < 640; p += SENSOR_GROUP_SIZE)
	{
		cudaMemset(dev_u, 0.0, size_u);

		int jp1 = jj[p];
		int jp2 = jj[p + SENSOR_GROUP_SIZE - 1];
		int ip1 = ii[p];
		int ip2 = ii[p + SENSOR_GROUP_SIZE - 1];

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
			propagation<<<Grid_Size, Block_Size>>>(jp1, jp2, ip1, ip2, dev_fo, dev_u, k);
		}

		// Four corners

		propagation_at_corners<<<1, NT>>>(dev_u);

		initial_signal<<<NT - 2, 159>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, p);
	}

	// Kaczmarz method
	// propagation

	/* float *f_t = new float[Nx_Ny * I_MAX]; */

	float *dev_rr1;
	int size_rr1 = Nx_Nt * sizeof(float);

	float *dev_rr2;
	int size_rr2 = Nx_Nt * sizeof(float);

	float *dev_rr3;
	int size_rr3 = Nx_Nt * sizeof(float);

	float *dev_rr4;
	int size_rr4 = Nx_Nt * sizeof(float);

	float *dev_z;
	int size_z = Nx_Ny * (NT + 1) * sizeof(float);

	float *dev_Lu;
	int size_Lu = Nx_Ny_Nt * sizeof(float);

	float *dev_f;
	int size_f = Nx_Ny * sizeof(float);

	float *dev_df;
	int size_df = Nx_Ny * sizeof(float);

	/* float *dev_alpha; */
	/* int size_alpha = Nx_Ny * sizeof(float); */

	float *dev_f_minus_fo;
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
	/* cudaMalloc((void**) &dev_alpha, size_alpha); */
	cudaMalloc((void**) &dev_f_minus_fo, size_f_minus_fo);

	cudaMemset(dev_rr1, 0.0, size_rr1);
	cudaMemset(dev_rr2, 0.0, size_rr2);
	cudaMemset(dev_rr3, 0.0, size_rr3);
	cudaMemset(dev_rr4, 0.0, size_rr4);
	cudaMemset(dev_f, 0.0, size_f);
	cudaMemset(dev_Lu, 0.0, size_Lu);

	float *f = new float[Nx_Ny];
	float *f_minus_fo = new float[Nx_Ny];
	float epsilon = 0.f;

	for(int iter = 0; iter < I_MAX; iter++)
	{
		printf("\nIter: %d\n", iter);
		cudaMemset(dev_u, 0.0, size_u);

		for(p = 0; p < 640; p += SENSOR_GROUP_SIZE)
		{
			int jp1 = jj[p];
			int jp2 = jj[p + SENSOR_GROUP_SIZE - 1];
			int ip1 = ii[p];
			int ip2 = ii[p + SENSOR_GROUP_SIZE - 1];

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
				propagation<<<Grid_Size, Block_Size>>>(jp1, jp2, ip1, ip2, dev_f, dev_u, k);
			}

			// Four corners

			propagation_at_corners<<<1, NT>>>(dev_u);
			difference_signal<<<NT - 2, 159>>>(dev_u, dev_g1, dev_g2, dev_g3, dev_g4, dev_rr1, dev_rr2, dev_rr3, dev_rr4, p);

			cudaMemset(dev_z, 0.0, size_z);

			for(k = NT - 2; k > 0; k--)
			{
				backpropagation1<<<Grid_Size, Block_Size>>>(dev_z, dev_f, k);
				backpropagation2<<<1, NX>>>(dev_z, dev_rr1, dev_rr2, dev_rr3, dev_rr4, k);
			}

            laplace1<<<dim3(25, 25, 50), dim3(8, 8, 8)>>>(dev_u, dev_Lu);
            laplace2<<<1, 1>>>(dev_u, dev_Lu);

			init_differential<<<Grid_Size, Block_Size>>>(dev_df, dev_z, dev_Lu, dev_f);

			for(k = 2; k < NT; k++)
			{
				update_differential<<<Grid_Size, Block_Size>>>(dev_df, dev_z, dev_Lu, dev_f, k);
			}

			/* update_field<<<Grid_Size, Block_Size>>>(dev_alpha, dev_f, dev_df, dev_f_minus_fo, dev_fo); */
			update_field<<<Grid_Size, Block_Size>>>(dev_f, dev_df, dev_f_minus_fo, dev_fo);
		}

		cudaMemcpy(f_minus_fo, dev_f_minus_fo, size_f_minus_fo, cudaMemcpyDeviceToHost);

		epsilon = norm(f_minus_fo, Nx_Ny) / norm(fo, Nx_Ny) * 100.f;

		printf("epsilon = %f\n", epsilon);

		if (epsilon < 20.f)
		{
			break;
		}
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
	/* cudaFree(dev_alpha); */
	cudaFree(dev_f_minus_fo);

	///////////////////////////

	/* float *image = new float[Nx_Ny]; */

	/* cudaMemcpy(image, dev_f, size_f, cudaMemcpyDeviceToHost); */

	/* ofstream file; */

	/* for(int yj = 0; yj < NY; yj++) */
	/* { */
	/* 	for(int xi = 0; xi < NX; xi++) */
	/* 	{ */
	/* 		file << image[xi + NX * yj]; */
	/* 		file << "\t"; */
	/* 	} */

	/* 	file << "\n"; */
	/* } */

	/* file.close(); */

	/////////////////////////
}

__global__ void field_setup(const float *x, const float *y, float *fo)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;
		float value = 0.f;

		/* if(((sqrtf(powf(x[i] - 0.015f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f) || ((sqrtf(powf(x[i] + 0.015f, 2.f) + powf(y[j] + 0.000f, 2.f))) <= 0.005f)) */
		/* { */
		/* 	value = 0.06f; */
		/* } */
		/* else */
		/* { */
		/* 	if(sqrtf(x[i] * x[i] + y[j] * y[j]) <= 0.03f) */
		/* 	{ */
		/* 		value = 0.02f; */
		/* 	} */
		/* 	else */
		/* 	{ */
		/* 		value = 0.f; */
		/* 	} */
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

	/* int NX * NY = NX * NY; */
	/* int NX * NY * k = NX * NY * k; */

	if((i < NX) && (j < NY))
	{
		/* int i + NX * j = i + NX * j; */

		float v = 1500.f * sqrtf(1.f + f[i + NX * j]);
		float r = v * dt / hx;
        float s = 2.f - 4.f * r * r;

        // at sensor
		if (((j + 1) >= jp1) && ((j + 1) <= jp2) && ((i + 1) >= ip1) && ((i + 1) <= ip2) && ((k + 1) <= 24))
		{
            float t = k * dt - tt;

			u[i + NX * j + NX * NY * (k + 1)] =
			    v * v * dt * dt *
			    cosf(omegac * t) *
			    expf(-(t * t) / (2.f * tao * tao)) +
			    r * r *
			    (u[(i + 1) + NX * j + NX * NY * k] +
			     u[(i - 1) + NX * j + NX * NY * k] +
			     u[i + NX * (j - 1) + NX * NY * k] +
			     u[i + NX * (j + 1) + NX * NY * k]) +
			    s * u[i + NX * j + NX * NY * k] -
			    u[i + NX * j + NX * NY * (k - 1)];
		}

        // not at sensor or boundary
		else if (i != 0 && j != 0 && i != (NX - 1) && j != (NY - 1))
		{
			u[i + NX * j + NX * NY * (k + 1)] =
			    r * r *
			    (u[(i + 1) + NX * j + NX * NY * k] +
			     u[(i - 1) + NX * j + NX * NY * k] +
			     u[i + NX * (j - 1) + NX * NY * k] +
			     u[i + NX * (j + 1) + NX * NY * k]) +
			    s * u[i + NX * j + NX * NY * k] -
			    u[i + NX * j + NX * NY * (k - 1)];
		}


        // left boundary
        else if ((i == 0) && (j > 0) && (j < (NY - 1)))
		{
			u[i + NX * j + NX * NY * (k + 1)] =
			    (2.f - r * (r + 2.f)) * u[i + NX * j + NX * NY * k] +
			    2.f * r * (1.f + r) * u[(i + NX * j + 1) + NX * NY * k] -
			    r * r * u[(i + NX * j + 2) + NX * NY * k] +
			    (2.f * r - 1.f) * u[i + NX * j + NX * NY * (k - 1)] -
			    2.f * r * u[(i + NX * j + 1) + NX * NY * (k - 1)];
		}


        // right boundary
        else if ((i == NX - 1) && (j > 0) && (j < (NY - 1)))
		{
			u[i + NX * j + NX * NY * (k + 1)] =
			    (2.f - 2.f * r - r * r) * u[i + NX * j + NX * NY * k] +
			    2.f * r * (1.f + r) * u[(i + NX * j - 1) + NX * NY * k] -
			    r * r * u[(i + NX * j - 2) + NX * NY * k] +
			    (2.f * r - 1.f) * u[i + NX * j + NX * NY * (k - 1)] -
			    2.f * r * u[(i + NX * j - 1) + NX * NY * (k - 1)];
		}

        // top boundary
        else if ((j == 0) && (i > 0) && (i < (NX - 1)))
		{
			u[i + NX * j + NX * NY * (k + 1)] =
			    (2.f - 2.f * r - r * r) * u[i + NX * j + NX * NY * k] +
			    2.f * r * (1.f + r) * u[(i + (j + 1) * NX) + NX * NY * k] -
			    r * r * u[(i + (j + 2) * NX) + NX * NY * k] +
			    (2.f * r - 1.f) * u[i + NX * j + NX * NY * (k - 1)] -
			    2.f * r * u[(i + (j + 1) * NX) + NX * NY * (k - 1)];
		}

        // bottom boundary
        else if ((j == NY - 1) && (i > 0) && (i < (NX - 1)))
		{
			u[i + NX * j + NX * NY * (k + 1)] =
			    (2.f - 2.f * r - r * r) * u[i + NX * j + NX * NY * k] +
			    2.f * r * (1.f + r) * u[(i + (j - 1) * NX) + NX * NY * k] -
			    r * r * u[(i + (j - 2) * NX) + NX * NY * k] +
			    (2.f * r - 1.f) * u[i + NX * j + NX * NY * (k - 1)] -
			    2.f * r * u[(i + (j - 1) * NX) + NX * NY * (k - 1)];
		}
	}
}

__global__ void propagation_at_corners(float *u)
{
	int k = threadIdx.x;
	int Nx_Ny = NX * NY;
	int Nx_Ny_k = Nx_Ny * k;

	u[Nx_Ny_k] =
	    1.f / 2.f * (u[NX + k] + u[1 + k]);

	u[(NX - 1) + Nx_Ny_k] =
	    1.f / 2.f * (u[(NX - 2) + Nx_Ny_k] + u[(NX - 1) + NX + Nx_Ny_k]);

	u[(NY - 1) * NX + Nx_Ny_k] =
	    1.f / 2.f * (u[(NY - 2) * NX + Nx_Ny_k] + u[1 +(NY - 1) * NX + Nx_Ny_k]);

	u[(NX - 1) + (NY - 1) * NX + Nx_Ny_k] =
	    1.f / 2.f * (u[(NX - 2) + (NY - 1) * NX + Nx_Ny_k] + u[(NX - 1) + (NY - 2) * NX + Nx_Ny_k]);

}

__global__ void initial_signal(const float *u, float *g1, float *g2, float *g3, float *g4, int p)
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

__global__ void difference_signal(const float *u, const float *g1, const float *g2, const float *g3, const float *g4, float *rr1, float *rr2, float *rr3, float *rr4, int p)
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

__global__ void backpropagation1(float *z, const float *f, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i >= 1) && (i < (NX - 1)) && (j >= 1) && (j < (NY - 1)))
	{
		z[i + NX * j + NX * NY * k] =
		    1500.f * 1500.f * (dt * dt) *
		    ((1.f + f[i + NX * (j - 1)]) * z[i + NX * (j - 1) + NX * NY * (k + 1)] +
		     (1.f + f[i + NX * (j + 1)]) * z[i + NX * (j + 1) + NX * NY * (k + 1)] +
		     (1.f + f[(i - 1) + NX * j]) * z[(i - 1) + NX * j + NX * NY * (k + 1)] +
		     (1.f + f[(i + 1) + NX * j]) * z[(i + 1) + NX * j + NX * NY * (k + 1)] -
		     4.f * (1.f + f[i + NX * j]) *
		     z[i + NX * j + NX * NY * (k + 1)]) / (h * h) +
		    2.f * z[i + NX * j + NX * NY * (k + 1)] -
		    z[i + NX * j + NX * NY * (k + 2)];
	}
}

__global__ void backpropagation2(float *z, const float *rr1, const float *rr2, const float *rr3, const float *rr4, int k)
{
	int i = threadIdx.x;

	if((i >= 21) && (i < 180))
	{
		z[i + NX * 180 + NX * NY * k] = 
		    z[i + NX * 179 + NX * NY * k] + 
		    rr1[i + NX * k] * h * 1000.f;

		z[i + NX * 20 + NX * NY * k] = 
		    z[i + NX * 21 + NX * NY * k] + 
		    rr3[i + NX * k] * h * 1000.f;

		z[180 + NX * i + NX * NY * k] = 
		    z[179 + NX * i + NX * NY * k] + 
		    rr2[i + NX * k] * h * 1000.f;

		z[20 + NX * i + NX * NY * k] = 
		    z[21 + NX * i + NX * NY * k] 
		    + rr4[i + NX * k] * h * 1000.f;
	}

	if((i >= 1) && (i < (NX - 1)))
	{
		z[i + NX * NY * k] = 
		    z[i + NX + NX * NY * k];

		z[i + NX * (NY - 1) + NX * NY * k] = 
		    z[i + NX * (NY - 2) + NX * NY * k];

		z[NX * i + NX * NY * k] = 
		    z[1 + NX * i + NX * NY * k];

		z[(NX - 1) + NX * i + NX * NY * k] = 
		    z[(NX - 2) + NX * i + NX * NY * k];
	}

    else if(i == 0)
	{
		z[NX * NY * k] = 
		    (z[1 + NX * NY * k] + 
		     z[NX + NX * NY * k]) / 2.f;

		z[(NX - 1) + NX * NY * k] = 
		    (z[(NX - 2) + NX * NY * k] + 
		     z[(NX - 1) + NX + NX * NY * k]) / 2.f;

		z[NX * (NY - 1) + NX * NY * k] = 
		    (z[1 + NX * (NY - 1) + NX * NY * k] + 
		     z[NX * (NY - 2) + NX * NY * k]) / 2.f;

		z[(NX - 1) + NX * (NY - 1) + NX * NY * k] = 
		    (z[(NX - 2) + NX * (NY - 1) + NX * NY * k] + 
		     z[(NX - 1) + NX * (NY - 2) + NX * NY * k]) / 2.f;
	}
}

__global__ void laplace1(const float *u, float *Lu)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

	if(i < (NX - 1) && j < (NY - 1) && k < NT)
	{
	    if (i >= 1) {
            if (j >= 1) {
                Lu[i + NX * j + NX * NY * k] =
                    (u[i + NX * (j - 1) + NX * NY * k] +
                    u[i + NX * (j + 1) + NX * NY * k] +
                    u[(i - 1) + NX * j + NX * NY * k] +
                    u[(i + 1) + NX * j + NX * NY * k] -
                    4.f * u[i + NX * j + NX * NY * k]) / (h * h);
            }

            else {
                Lu[i + NX * NY * k] =
                    (u[i + NX * NY * k] +
                    u[i + NX + NX * NY * k] +
                    u[(i - 1) + NX * NY * k] +
                    u[(i + 1) + NX * NY * k] -
                    4.f * u[i + NX * NY * k]) / (h * h);

                Lu[i + NX * (NY - 1) + NX * NY * k] =
                    (u[i + NX * (NY - 1) + NX * NY * k] +
                    u[i + NX * (NY - 2) + NX * NY * k] +
                    u[(i - 1) + NX * (NY - 1) + NX * NY * k] +
                    u[(i + 1) + NX * (NY - 1) + NX * NY * k] -
                    4.f * u[i + NX * (NY - 1) + NX * NY * k]) / (h * h);

                Lu[NX * i + NX * NY * k] =
                    (u[NX * i + NX * NY * k] +
                    u[1 + NX * i + NX * NY * k] +
                    u[NX * (i - 1) + NX * NY * k] +
                    u[NX * (i + 1) + NX * NY * k] -
                    4.f * u[NX * i + NX * NY * k]) / (h * h);

                Lu[(NX - 1) + NX * i + NX * NY * k] =
                    (u[(NX - 1) + NX * i + NX * NY * k] +
                    u[(NX - 2) + NX * i + NX * NY * k] +
                    u[(NX - 1) + NX * (i - 1) + NX * NY * k] +
                    u[(NX - 1) + NX * (i + 1) + NX * NY * k] -
                    4.f * u[(NX - 1) + NX * i + NX * NY * k]) / (h * h);
            }
        }
	}
}

__global__ void laplace2(const float *u, float *Lu)
{
    # pragma unroll
    for (int k = 1; k < NT; ++k) {
        Lu[NX * NY * k] =
            (Lu[1 + NX * NY * k] +
            Lu[NX + NX * NY * k]) / 2.f;

        Lu[(NX - 1) + NX * NY * k] =
            (Lu[(NX - 2) + NX * NY * k] +
            Lu[(NX - 1) + NX + NX * NY * k]) / 2.f;

        Lu[NX * (NY - 1) + NX * NY * k] =
            (Lu[1 + NX * (NY - 1) + NX * NY * k] +
            Lu[NX * (NY - 2) + NX * NY * k]) / 2.f;

        Lu[(NX - 1) + NX * (NY - 1) + NX * NY * k] =
            (Lu[(NX - 2) + NX * (NY - 1) + NX * NY * k] +
            Lu[(NX - 1) + NX * (NY - 2) + NX * NY * k]) / 2.f;
    }
}

__global__ void init_differential(float *df, const float *z, const float *Lu, const float *f)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int Nx_Ny = NX * NY;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		df[offset] = z[offset + Nx_Ny] * Lu[offset + Nx_Ny] / (1.f + f[offset]);
	}
}

__global__ void update_differential(float *df, const float *z, const float *Lu, const float *f, int k)
{
	// Map from threadIdx / BlockIdx to pixel position

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY)) {
		df[i + NX * j] += z[i + NX * j + NX * NY * k] * Lu[i + NX * j + NX * NY * k] / (1.f + f[i + NX * j]);
	}
}

/* __global__ void update_field(float *alpha, float *f, const float *df, float *f_minus_fo, const float *fo) */
__global__ void update_field(float *f, const float *df, float *f_minus_fo, const float *fo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

        bool flag = (i >= 21) && (i < 180) && (j >= 21) && (j < 180);

        float alpha = flag ? 1.f : 0.f;

		f[offset] += 20000.f * alpha * df[offset];
		f_minus_fo[offset] = f[offset] - fo[offset];
	}
}

__global__ void reset(const float *f, float *v, float *r, float *r2, float *s)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i < NX) && (j < NY))
	{
		int offset = i + NX * j;

		v[offset] = 1500.f * sqrtf(1.f + f[offset]);
		r[offset] = v[offset] * dt / hx;
		r2[offset] = r[offset] * r[offset];
		s[offset] = 2.f - 4.f * r2[offset];
	}
}

void IO_Files(float *x, float *y, float *fo, float *f)
{
	int i = 0, j = 0;
	/* int k = 0; */

	// I/O Files

	ofstream x_file, y_file;
	ofstream fo_file;
	ofstream f_file;

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
			fo_file << fo[i + NX * j];
			fo_file << " ";
		}

		fo_file << "\n";
	}

	for(j = 0; j < NY; j++)
	{
		for(i = 0; i < NX; i++)
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

float norm(float *A, int length)
{
	float sum = 0;

	for(int i = 0; i < length; i++) {
		sum += A[i] * A[i];
	}

	return sqrtf(sum);
}
