#ifndef CUDA_PTR_CUH
#define CUDA_PTR_CUH
#include <cassert>

// Wrapper class for a pointer to host memory that resides in host memory
// A host_ptr may or may not be responsible for allocating and deallocating its own memory
template <typename T>
struct host_ptr
{
	// host_ptr constructor taking arguments for size
	// a host_ptr constructed with this constructor
	// is responsible for allocating and deallocating its own memory
	explicit host_ptr(int nx, int ny = 1, int nz = 1, int nw = 1)
		: ptr{new T[nx * ny * nz * nw]}
		, dim{nx, ny, nz, nw}
		, own{true}
	{}

	// host_ptr to host_ptr copy constructor
	// a host_ptr that is copied from another host_ptr
	// is not responsible for its own memory
	// NOTE: if a host_ptr that was copy-constructed
	// outlives the original host_ptr, it will cause a memory leak
	host_ptr(const host_ptr<T> &h_ptr)
		: ptr{h_ptr.ptr}
		, dim(h_ptr.dim)
		, own{false}
	{}

	// destructor for host_ptr
	~host_ptr()
	{
		// if this host_ptr owns its memory,
		// then deallocate it when destructed
		if (own)
			delete [] ptr;
	}

	// call operator overload
	// allows indexing by A(i, j, k, l)
	T& operator()(int i, int j = 0, int k = 0, int l = 0)
	{
#ifdef DEBUG
		// array bounds checking
		assert(i >= 0);
		assert(i < dim.x);
		assert(j >= 0);
		assert(j < dim.y);
		assert(k >= 0);
		assert(k < dim.z);
		assert(l >= 0);
		assert(l < dim.w);
#endif
		return ptr[i + dim.x * (j + dim.y * (k + dim.z * l))];
	}

	// const version of call operator
	const T& operator()(int i, int j = 0, int k = 0, int l = 0) const
	{
#ifdef DEBUG
		// array bounds checking
		assert(i >= 0);
		assert(i < dim.x);
		assert(j >= 0);
		assert(j < dim.y);
		assert(k >= 0);
		assert(k < dim.z);
		assert(l >= 0);
		assert(l < dim.w);
#endif
		return ptr[i + dim.x * (j + dim.y * (k + dim.z * l))];
	}

	T* ptr; // pointer to host memory
	const int4 dim; // dimensions of array
	bool own; // boolean signaling whether it is responsible for its own memory
};

// Wrapper class for a pointer to device memory that resides in host memory
// A device_ptr is always responsible for allocating and deallocating its own memory
template <typename T>
struct device_ptr
{
	// device_ptr constructor taking arguments for size
	explicit device_ptr(int nx, int ny = 1, int nz = 1, int nw = 1)
		: dim{nx, ny, nz, nw}
	{
		cudaMalloc(&ptr, dim.x * dim.y * dim.z * dim.w * sizeof(T));
		set(T{});
	}

	// device_ptr destructor
	~device_ptr()
	{
		cudaFree(ptr);
	}

	// wrapper function for cudaMemset()
	void set(T val)
	{
		cudaMemset(ptr, val, dim.x * dim.y * dim.z * dim.w * sizeof(T));
	}

	T* ptr; // pointer to device memory
	const int4 dim; // dimensions of array
};

// Wrapper class for a pointer to device memory that resides in device memory
// A kernel_ptr is not responsible for allocating and deallocating its own memory,
// it always shares its memory with a device_ptr
template <typename T>
struct kernel_ptr
{
	// device_ptr to kernel_ptr copy constructor
	__device__
	kernel_ptr(const device_ptr<T> &d_ptr)
		: ptr{d_ptr.ptr}
		, dim(d_ptr.dim)
	{}

	// kernel_ptr to kernel_ptr copy constructor
	__device__
	kernel_ptr(const kernel_ptr<T> &) = default;

	// call operator overload
	// allows indexing by A(i, j, k, l)
	__device__
	T& operator()(int i, int j = 0, int k = 0, int l = 0)
	{
#ifdef DEBUG
		// array bounds checking
		assert(i >= 0);
		assert(i < dim.x);
		assert(j >= 0);
		assert(j < dim.y);
		assert(k >= 0);
		assert(k < dim.z);
		assert(l >= 0);
		assert(l < dim.w);
#endif
		return ptr[i + dim.x * (j + dim.y * (k + dim.z * l))];
	}

	// const version of call operator
	__device__
	const T& operator()(int i, int j = 0, int k = 0, int l = 0) const
	{
#ifdef DEBUG
		// array bounds checking
		assert(i >= 0);
		assert(i < dim.x);
		assert(j >= 0);
		assert(j < dim.y);
		assert(k >= 0);
		assert(k < dim.z);
		assert(l >= 0);
		assert(l < dim.w);
#endif
		return ptr[i + dim.x * (j + dim.y * (k + dim.z * l))];
	}

	T* ptr; // pointer to device memory
	const int4 dim; // dimensions of array
};

// wrapper functions for cudaMemcpy()
// allows copying between device_ptr and host_ptr

// in general, copy from source to destination:
// copy(destination, source)


// copy from d_ptr to h_ptr
template <typename T>
void copy(host_ptr<T> &h_ptr, device_ptr<T> const &d_ptr)
{
#ifdef DEBUG
	// check that both arguments have same dimensions
		assert(h_ptr.dim.x == d_ptr.dim.x);
		assert(h_ptr.dim.y == d_ptr.dim.y);
		assert(h_ptr.dim.z == d_ptr.dim.z);
		assert(h_ptr.dim.w == d_ptr.dim.w);
#endif
	const int4 &dim = h_ptr.dim;
	cudaMemcpy(h_ptr.ptr, d_ptr.ptr, dim.x * dim.y * dim.z * dim.w * sizeof(T), cudaMemcpyDeviceToHost);
}

// copy from h_ptr to d_ptr
template <typename T>
void copy(device_ptr<T> &d_ptr, host_ptr<T> const &h_ptr)
{
#ifdef DEBUG
	// check that both arguments have same dimensions
		assert(h_ptr.dim.x == d_ptr.dim.x);
		assert(h_ptr.dim.y == d_ptr.dim.y);
		assert(h_ptr.dim.z == d_ptr.dim.z);
		assert(h_ptr.dim.w == d_ptr.dim.w);
#endif
	const int4 &dim = d_ptr.dim;
	cudaMemcpy(d_ptr.ptr, h_ptr.ptr, dim.x * dim.y * dim.z * dim.w * sizeof(T), cudaMemcpyHostToDevice);
}

// copy from d_ptr to d0_ptr
template <typename T>
void copy(device_ptr<T> &d0_ptr, device_ptr<T> const &d_ptr)
{
#ifdef DEBUG
	// check that both arguments have same dimensions
		assert(d_ptr.dim.x == d0_ptr.dim.x);
		assert(d_ptr.dim.y == d0_ptr.dim.y);
		assert(d_ptr.dim.z == d0_ptr.dim.z);
		assert(d_ptr.dim.w == d0_ptr.dim.w);
#endif
	const int4 &dim = d0_ptr.dim;
	cudaMemcpy(d0_ptr.ptr, d_ptr.ptr, dim.x * dim.y * dim.z * dim.w * sizeof(T), cudaMemcpyDeviceToDevice);
}

// copy from h_ptr to h0_ptr
template <typename T>
void copy(host_ptr<T> &h0_ptr, host_ptr<T> const &h_ptr)
{
#ifdef DEBUG
	// check that both arguments have same dimensions
		assert(h_ptr.dim.x == h0_ptr.dim.x);
		assert(h_ptr.dim.y == h0_ptr.dim.y);
		assert(h_ptr.dim.z == h0_ptr.dim.z);
		assert(h_ptr.dim.w == h0_ptr.dim.w);
#endif
	const int4 &dim = h0_ptr.dim;
	cudaMemcpy(h0_ptr.ptr, h_ptr.ptr, dim.x * dim.y * dim.z * dim.w * sizeof(T), cudaMemcpyHostToHost);
}

#endif // CUDA_PTR_CUH
