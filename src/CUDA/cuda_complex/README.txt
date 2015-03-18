This is an implementation of C++ std::complex for use on CUDA devices.
Written by John C. Travers <jtravs@gmail.com> (2012).

Apart from nvcc, it should also work wih any C++03 compiler.
It is quiet complete. As far as I can tell the only missing features are:
  - long double support (not supported on CUDA)
  - some integral pow functions (due to lack of C++11 support on CUDA)

This code is heavily derived from the LLVM libcpp project
(svn revision 147853), mainly libcxx/include/complex. The git history
contains the complete change history from the original.

The modifications are licensed as per the original LLVM license, which is dual
licensed under the MIT and the University of Illinois Open Source Licenses.
See LICENSE.TXT for details.
