#ifndef MIMO_IO_CUH
#define MIMO_IO_CUH
#include <ios>
#include <limits>

template <class Stream, typename T>
void read(Stream &str, host_ptr<T> h_ptr)
{
	const int4 &dim = h_ptr.dim;

	for (int l = 0; l < dim.w; ++l) {
		for (int k = 0; k < dim.z; ++k) {
			for (int j = 0; j < dim.y; ++j) {
				for (int i = 0; i < dim.x; ++i) {
					str >> h_ptr(i, j, k, l);
				}
			}
		}
	}
}

template <class Stream, typename T>
void write(Stream &str, host_ptr<T> const h_ptr)
{
	auto flags(str.flags());
	str.unsetf(std::ios_base::floatfield);
	str.precision(std::numeric_limits<T>::max_digits10);

	const int4 &dim = h_ptr.dim;

	for (int l = 0; l < dim.w; ++l) {
		for (int k = 0; k < dim.z; ++k) {
			for (int j = 0; j < dim.y; ++j) {
				for (int i = 0; i < dim.x; ++i) {
					str << h_ptr(i, j, k, l) << " ";
				}
				str << "\n";
			}
			str << "\n";
		}
		str << "\n";
	}

	str.flags(flags);
}

#endif // MIMO_IO_CUH
