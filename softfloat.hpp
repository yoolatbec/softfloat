// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from package issued with the following license:

/*============================================================================

 This C header file is part of the SoftFloat IEEE Floating-Point Arithmetic
 Package, Release 3c, by John R. Hauser.

 Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
 University of California.  All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
 this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions, and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
 be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 =============================================================================*/

#pragma once
#ifndef softfloat_h
#define softfloat_h 1

#include "cvdef.h"
#include <climits>

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define __HOST_DEVICE__ __host__ __device__
#define __HOST__ __host__

#elif defined(SYCL)

#include <sycl/sycl.hpp>
#define __HOST_DEVICE__ SYCL_EXTERNAL

#else
#define __HOST_DEVICE__
#define __HOST__
#endif

#ifndef CV_BIG_INT
#define CV_BIG_INT(x) ((long)x)
#endif

namespace cv {

/** @addtogroup core_utils_softfloat

 [SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) is a software implementation
 of floating-point calculations according to IEEE 754 standard.
 All calculations are done in integers, that's why they are machine-independent and bit-exact.
 This library can be useful in accuracy-critical parts like look-up tables generation, tests, etc.
 OpenCV contains a subset of SoftFloat partially rewritten to C++.

 ### Types

 There are two basic types: @ref softfloat and @ref softdouble.
 These types are binary compatible with float and double types respectively
 and support conversions to/from them.
 Other types from original SoftFloat library like fp16 or fp128 were thrown away
 as well as quiet/signaling NaN support, on-the-fly rounding mode switch
 and exception flags (though exceptions can be implemented in the future).

 ### Operations

 Both types support the following:
 - Construction from signed and unsigned 32-bit and 64 integers,
 float/double or raw binary representation
 - Conversions between each other, to float or double and to int
 using @ref cvRound, @ref cvTrunc, @ref cvFloor, @ref cvCeil or a bunch of
 saturate_cast functions
 - Add, subtract, multiply, divide, remainder, square root, FMA with absolute precision
 - Comparison operations
 - Explicit sign, exponent and significand manipulation through get/set methods,
 number state indicators (isInf, isNan, isSubnormal)
 - Type-specific constants like eps, minimum/maximum value, best pi approximation, etc.
 - min(), max(), abs(), exp(), log() and pow() functions

 */
//! @{
struct softfloat;
struct softdouble;

#ifdef CUDA
void initConstants();
#endif

struct softfloat {
public:
	/** @brief Default constructor */
//	__HOST_DEVICE__ softfloat() {
//		v = 0;
//	}
	/** @brief Copy constructor */
//	__HOST__ softfloat(const softfloat &c) {
//		v = c.v;
//	}
	/** @brief Assign constructor */
	__HOST_DEVICE__ softfloat& operator=(const softfloat &c) {
		if (&c != this)
			v = c.v;
		return *this;
	}
	/** @brief Construct from raw

	 Builds new value from raw binary representation
	 */
	__HOST_DEVICE__ static const softfloat fromRaw(const uint32_t a) {
		softfloat x;
		x.v = a;
		return x;
	}

	/** @brief Construct from integer */
//	__HOST_DEVICE__ explicit softfloat(const uint32_t);
//	__HOST_DEVICE__ explicit softfloat(const uint64_t);
//	__HOST_DEVICE__ explicit softfloat(const int32_t);
//	__HOST_DEVICE__ explicit softfloat(const int64_t);
	__HOST_DEVICE__ static softfloat fromUInt32(uint32_t);
	__HOST_DEVICE__ static softfloat fromUInt64(uint64_t);
	__HOST_DEVICE__ static softfloat fromInt32(int32_t);
	__HOST_DEVICE__ static softfloat fromInt64(int64_t);
	__HOST_DEVICE__ static softfloat fromFloat(float a) {
		Cv32suf s;
		softfloat r;
		s.f = a;
		r.v = s.u;
		return r;
	}

#ifdef CV_INT32_T_IS_LONG_INT
    // for platforms with int32_t = long int
	__HOST_DEVICE__ explicit softfloat( const int a ) { *this = softfloat(static_cast<int32_t>(a)); }
#endif

	/** @brief Construct from float */
//	__HOST_DEVICE__ explicit softfloat(const float a) {
//		Cv32suf s;
//		s.f = a;
//		v = s.u;
//	}
	/** @brief Type casts  */
	__HOST_DEVICE__ operator softdouble() const;
	__HOST_DEVICE__ operator float() const {
		Cv32suf s;
		s.u = v;
		return s.f;
	}

	/** @brief Basic arithmetics */
	__HOST_DEVICE__ softfloat operator +(const softfloat&) const;
	__HOST_DEVICE__ softfloat operator -(const softfloat&) const;
	__HOST_DEVICE__ softfloat operator *(const softfloat&) const;
	__HOST_DEVICE__ softfloat operator /(const softfloat&) const;
	__HOST_DEVICE__ softfloat operator -() const {
		softfloat x;
		x.v = v ^ (1U << 31);
		return x;
	}

	/** @brief Remainder operator

	 A quote from original SoftFloat manual:

	 > The IEEE Standard remainder operation computes the value
	 > a - n * b, where n is the integer closest to a / b.
	 > If a / b is exactly halfway between two integers, n is the even integer
	 > closest to a / b. The IEEE Standard’s remainder operation is always exact and so requires no rounding.
	 > Depending on the relative magnitudes of the operands, the remainder functions
	 > can take considerably longer to execute than the other SoftFloat functions.
	 > This is an inherent characteristic of the remainder operation itself and is not a flaw
	 > in the SoftFloat implementation.
	 */
	__HOST_DEVICE__ softfloat operator %(const softfloat&) const;

	__HOST_DEVICE__ softfloat& operator +=(const softfloat &a) {
		*this = *this + a;
		return *this;
	}
	__HOST_DEVICE__ softfloat& operator -=(const softfloat &a) {
		*this = *this - a;
		return *this;
	}
	__HOST_DEVICE__ softfloat& operator *=(const softfloat &a) {
		*this = *this * a;
		return *this;
	}
	__HOST_DEVICE__ softfloat& operator /=(const softfloat &a) {
		*this = *this / a;
		return *this;
	}
	__HOST_DEVICE__ softfloat& operator %=(const softfloat &a) {
		*this = *this % a;
		return *this;
	}

	/** @brief Comparison operations

	 - Any operation with NaN produces false
	 + The only exception is when x is NaN: x != y for any y.
	 - Positive and negative zeros are equal
	 */
	__HOST_DEVICE__ bool operator ==(const softfloat&) const;
	__HOST_DEVICE__ bool operator !=(const softfloat&) const;
	__HOST_DEVICE__ bool operator >(const softfloat&) const;
	__HOST_DEVICE__ bool operator >=(const softfloat&) const;
	__HOST_DEVICE__ bool operator <(const softfloat&) const;
	__HOST_DEVICE__ bool operator <=(const softfloat&) const;

	/** @brief NaN state indicator */
	__HOST_DEVICE__ inline bool isNaN() const {
		return (v & 0x7fffffff) > 0x7f800000;
	}
	/** @brief Inf state indicator */
	__HOST_DEVICE__ inline bool isInf() const {
		return (v & 0x7fffffff) == 0x7f800000;
	}
	/** @brief Subnormal number indicator */
	__HOST_DEVICE__ inline bool isSubnormal() const {
		return ((v >> 23) & 0xFF) == 0;
	}

	/** @brief Get sign bit */
	__HOST_DEVICE__ inline bool getSign() const {
		return (v >> 31) != 0;
	}
	/** @brief Construct a copy with new sign bit */
	__HOST_DEVICE__ inline softfloat setSign(bool sign) const {
		softfloat x;
		x.v = (v & ((1U << 31) - 1)) | ((uint32_t) sign << 31);
		return x;
	}
	/** @brief Get 0-based exponent */
	__HOST_DEVICE__ inline int getExp() const {
		return ((v >> 23) & 0xFF) - 127;
	}
	/** @brief Construct a copy with new 0-based exponent */
	__HOST_DEVICE__ inline softfloat setExp(int e) const {
		softfloat x;
		x.v = (v & 0x807fffff) | (((e + 127) & 0xFF) << 23);
		return x;
	}

	/** @brief Get a fraction part

	 Returns a number 1 <= x < 2 with the same significand
	 */
	__HOST_DEVICE__ inline softfloat getFrac() const {
		uint_fast32_t vv = (v & 0x007fffff) | (127 << 23);
		return softfloat::fromRaw(vv);
	}
	/** @brief Construct a copy with provided significand

	 Constructs a copy of a number with significand taken from parameter
	 */
	__HOST_DEVICE__ inline softfloat setFrac(const softfloat &s) const {
		softfloat x;
		x.v = (v & 0xff800000) | (s.v & 0x007fffff);
		return x;
	}

	/** @brief Zero constant */
	__HOST_DEVICE__ static softfloat zero() {
		return softfloat::fromRaw(0);
	}
	/** @brief Positive infinity constant */
	__HOST_DEVICE__ static softfloat inf() {
		return softfloat::fromRaw(0xFF << 23);
	}
	/** @brief Default NaN constant */
	__HOST_DEVICE__ static softfloat nan() {
		return softfloat::fromRaw(0x7fffffff);
	}
	/** @brief One constant */
	__HOST_DEVICE__ static softfloat one() {
		return softfloat::fromRaw(127 << 23);
	}
	/** @brief Smallest normalized value */
	__HOST_DEVICE__ static softfloat min() {
		return softfloat::fromRaw(0x01 << 23);
	}
	/** @brief Difference between 1 and next representable value */
	__HOST_DEVICE__ static softfloat eps() {
		return softfloat::fromRaw((127 - 23) << 23);
	}
	/** @brief Biggest finite value */
	__HOST_DEVICE__ static softfloat max() {
		return softfloat::fromRaw((0xFF << 23) - 1);
	}
	/** @brief Correct pi approximation */
	__HOST_DEVICE__ static softfloat pi() {
		return softfloat::fromRaw(0x40490fdb);
	}

	uint32_t v = 0;
};

/*----------------------------------------------------------------------------
 *----------------------------------------------------------------------------*/

struct softdouble {
public:
	/** @brief Default constructor */
//	__HOST_DEVICE__ softdouble()
//			: v(0) {
//	}
	/** @brief Copy constructor */
//	__HOST_DEVICE__ softdouble(const softdouble &c) {
//		v = c.v;
//	}
	/** @brief Assign constructor */
	__HOST_DEVICE__ softdouble& operator=(const softdouble &c) {
		if (&c != this)
			v = c.v;
		return *this;
	}
	/** @brief Construct from raw

	 Builds new value from raw binary representation
	 */
	__HOST_DEVICE__ static softdouble fromRaw(const uint64_t a) {
		softdouble x;
		x.v = a;
		return x;
	}

	/** @brief Construct from integer */
//	__HOST_DEVICE__ explicit softdouble(const uint32_t);
//	__HOST_DEVICE__ explicit softdouble(const uint64_t);
//	__HOST_DEVICE__ explicit softdouble(const int32_t);
//	__HOST_DEVICE__ explicit softdouble(const int64_t);
	__HOST_DEVICE__ static softdouble fromUInt32(uint32_t);
	__HOST_DEVICE__ static softdouble fromUInt64(uint64_t);
	__HOST_DEVICE__ static softdouble fromInt32(int32_t);
	__HOST_DEVICE__ static softdouble fromInt64(int64_t);

#ifdef CV_INT32_T_IS_LONG_INT
	// for platforms with int32_t = long int
	explicit softdouble( const int a ) {*this = softdouble(static_cast<int32_t>(a));}
#endif

	/** @brief Construct from double */
//	__HOST_DEVICE__ explicit softdouble(const double a) {
//		Cv64suf s;
//		s.f = a;
//		v = s.u;
//	}
	__HOST_DEVICE__ static softdouble fromDouble(double a) {
		softdouble r;
		Cv64suf s;
		s.f = a;
		r.v = s.u;
		return r;
	}

	/** @brief Type casts  */
	__HOST_DEVICE__ operator softfloat() const;
	__HOST_DEVICE__ operator double() const {
		Cv64suf s;
		s.u = v;
		return s.f;
	}

	/** @brief Basic arithmetics */
	__HOST_DEVICE__ softdouble operator +(const softdouble&) const;
	__HOST_DEVICE__ softdouble operator -(const softdouble&) const;
	__HOST_DEVICE__ softdouble operator *(const softdouble&) const;
	__HOST_DEVICE__ softdouble operator /(const softdouble&) const;
	__HOST_DEVICE__ softdouble operator -() const {
		softdouble x;
		x.v = v ^ (1ULL << 63);
		return x;
	}

	/** @brief Remainder operator

	 A quote from original SoftFloat manual:

	 > The IEEE Standard remainder operation computes the value
	 > a - n * b, where n is the integer closest to a / b.
	 > If a / b is exactly halfway between two integers, n is the even integer
	 > closest to a / b. The IEEE Standard’s remainder operation is always exact and so requires no rounding.
	 > Depending on the relative magnitudes of the operands, the remainder functions
	 > can take considerably longer to execute than the other SoftFloat functions.
	 > This is an inherent characteristic of the remainder operation itself and is not a flaw
	 > in the SoftFloat implementation.
	 */
	__HOST_DEVICE__ softdouble operator %(const softdouble&) const;

	__HOST_DEVICE__ softdouble& operator +=(const softdouble &a) {
		*this = *this + a;
		return *this;
	}
	__HOST_DEVICE__ softdouble& operator -=(const softdouble &a) {
		*this = *this - a;
		return *this;
	}
	__HOST_DEVICE__ softdouble& operator *=(const softdouble &a) {
		*this = *this * a;
		return *this;
	}
	__HOST_DEVICE__ softdouble& operator /=(const softdouble &a) {
		*this = *this / a;
		return *this;
	}
	__HOST_DEVICE__ softdouble& operator %=(const softdouble &a) {
		*this = *this % a;
		return *this;
	}

	/** @brief Comparison operations

	 - Any operation with NaN produces false
	 + The only exception is when x is NaN: x != y for any y.
	 - Positive and negative zeros are equal
	 */
	__HOST_DEVICE__ bool operator ==(const softdouble&) const;
	__HOST_DEVICE__ bool operator !=(const softdouble&) const;
	__HOST_DEVICE__ bool operator >(const softdouble&) const;
	__HOST_DEVICE__ bool operator >=(const softdouble&) const;
	__HOST_DEVICE__ bool operator <(const softdouble&) const;
	__HOST_DEVICE__ bool operator <=(const softdouble&) const;

	/** @brief NaN state indicator */
	__HOST_DEVICE__ inline bool isNaN() const {
		return (v & 0x7fffffffffffffff) > 0x7ff0000000000000;
	}
	/** @brief Inf state indicator */
	__HOST_DEVICE__ inline bool isInf() const {
		return (v & 0x7fffffffffffffff) == 0x7ff0000000000000;
	}
	/** @brief Subnormal number indicator */
	__HOST_DEVICE__ inline bool isSubnormal() const {
		return ((v >> 52) & 0x7FF) == 0;
	}

	/** @brief Get sign bit */
	__HOST_DEVICE__ inline bool getSign() const {
		return (v >> 63) != 0;
	}
	/** @brief Construct a copy with new sign bit */
	__HOST_DEVICE__ softdouble setSign(bool sign) const {
		softdouble x;
		x.v = (v & ((1ULL << 63) - 1)) | ((uint_fast64_t) (sign) << 63);
		return x;
	}
	/** @brief Get 0-based exponent */
	__HOST_DEVICE__ inline int getExp() const {
		return ((v >> 52) & 0x7FF) - 1023;
	}
	/** @brief Construct a copy with new 0-based exponent */
	__HOST_DEVICE__ inline softdouble setExp(int e) const {
		softdouble x;
		x.v = (v & 0x800FFFFFFFFFFFFF)
				| ((uint_fast64_t) ((e + 1023) & 0x7FF) << 52);
		return x;
	}

	/** @brief Get a fraction part

	 Returns a number 1 <= x < 2 with the same significand
	 */
	__HOST_DEVICE__ inline softdouble getFrac() const {
		uint_fast64_t vv = (v & 0x000FFFFFFFFFFFFF)
				| ((uint_fast64_t) (1023) << 52);
		return softdouble::fromRaw(vv);
	}
	/** @brief Construct a copy with provided significand

	 Constructs a copy of a number with significand taken from parameter
	 */
	__HOST_DEVICE__ inline softdouble setFrac(const softdouble &s) const {
		softdouble x;
		x.v = (v & 0xFFF0000000000000) | (s.v & 0x000FFFFFFFFFFFFF);
		return x;
	}

	/** @brief Zero constant */
	__HOST_DEVICE__ static softdouble zero() {
		return softdouble::fromRaw(0);
	}
	/** @brief Positive infinity constant */
	__HOST_DEVICE__ static softdouble inf() {
		return softdouble::fromRaw((uint_fast64_t) (0x7FF) << 52);
	}
	/** @brief Default NaN constant */
	__HOST_DEVICE__ static softdouble nan() {
		return softdouble::fromRaw(CV_BIG_INT(0x7FFFFFFFFFFFFFFF));
	}
	/** @brief One constant */
	__HOST_DEVICE__ static softdouble one() {
		return softdouble::fromRaw((uint_fast64_t) (1023) << 52);
	}
	/** @brief Smallest normalized value */
	__HOST_DEVICE__ static softdouble min() {
		return softdouble::fromRaw((uint_fast64_t) (0x01) << 52);
	}
	/** @brief Difference between 1 and next representable value */
	__HOST_DEVICE__ static softdouble eps() {
		return softdouble::fromRaw((uint_fast64_t) (1023 - 52) << 52);
	}
	/** @brief Biggest finite value */
	__HOST_DEVICE__ static softdouble max() {
		return softdouble::fromRaw(((uint_fast64_t) (0x7FF) << 52) - 1);
	}
	/** @brief Correct pi approximation */
	__HOST_DEVICE__ static softdouble pi() {
		return softdouble::fromRaw(CV_BIG_INT(0x400921FB54442D18));
	}

	uint64_t v = 0;
};

/*----------------------------------------------------------------------------
 *----------------------------------------------------------------------------*/

/** @brief Fused Multiplication and Addition

 Computes (a*b)+c with single rounding
 */
__HOST_DEVICE__ softfloat mulAdd(const softfloat &a, const softfloat &b,
		const softfloat &c);
__HOST_DEVICE__ softdouble mulAdd(const softdouble &a, const softdouble &b,
		const softdouble &c);

/** @brief Square root */
__HOST_DEVICE__ softfloat sqrt(const softfloat &a);
__HOST_DEVICE__ softdouble sqrt(const softdouble &a);

}

/*----------------------------------------------------------------------------
 | Ported from OpenCV and added for usability
 *----------------------------------------------------------------------------*/

/** @brief Truncates number to integer with minimum magnitude */
__HOST_DEVICE__ int cvTrunc(const cv::softfloat &a);
__HOST_DEVICE__ int cvTrunc(const cv::softdouble &a);

/** @brief Rounds a number to nearest even integer */
__HOST_DEVICE__ int cvRound(const cv::softfloat &a);
__HOST_DEVICE__ int cvRound(const cv::softdouble &a);

/** @brief Rounds a number to nearest even long long integer */
__HOST_DEVICE__ int64_t cvRound64(const cv::softdouble &a);

/** @brief Rounds a number down to integer */
__HOST_DEVICE__ int cvFloor(const cv::softfloat &a);
__HOST_DEVICE__ int cvFloor(const cv::softdouble &a);

/** @brief Rounds number up to integer */
__HOST_DEVICE__ int cvCeil(const cv::softfloat &a);
__HOST_DEVICE__ int cvCeil(const cv::softdouble &a);

namespace cv {
/** @brief Saturate casts */
template<typename _Tp> static inline _Tp saturate_cast(softfloat a) {
	return _Tp(a);
}
template<typename _Tp> static inline _Tp saturate_cast(softdouble a) {
	return _Tp(a);
}

//__HOST_DEVICE__ template<> inline uchar saturate_cast<uchar>(softfloat a) {
//	return (uchar) std::max(std::min(cvRound(a), (int) UCHAR_MAX), 0);
//}
//__HOST_DEVICE__ template<> inline uchar saturate_cast<uchar>(softdouble a) {
//	return (uchar) std::max(std::min(cvRound(a), (int) UCHAR_MAX), 0);
//}
//
//__HOST_DEVICE__ template<> inline schar saturate_cast<schar>(softfloat a) {
//	return (schar) std::min(std::max(cvRound(a), (int) SCHAR_MIN),
//			(int) SCHAR_MAX);
//}
//__HOST_DEVICE__ template<> inline schar saturate_cast<schar>(softdouble a) {
//	return (schar) std::min(std::max(cvRound(a), (int) SCHAR_MIN),
//			(int) SCHAR_MAX);
//}
//
//__HOST_DEVICE__ template<> inline ushort saturate_cast<ushort>(softfloat a) {
//	return (ushort) std::max(std::min(cvRound(a), (int) USHRT_MAX), 0);
//}
//__HOST_DEVICE__ template<> inline ushort saturate_cast<ushort>(softdouble a) {
//	return (ushort) std::max(std::min(cvRound(a), (int) USHRT_MAX), 0);
//}
//
//__HOST_DEVICE__ template<> inline short saturate_cast<short>(softfloat a) {
//	return (short) std::min(std::max(cvRound(a), (int) SHRT_MIN),
//			(int) SHRT_MAX);
//}
//__HOST_DEVICE__ template<> inline short saturate_cast<short>(softdouble a) {
//	return (short) std::min(std::max(cvRound(a), (int) SHRT_MIN),
//			(int) SHRT_MAX);
//}
//
//__HOST_DEVICE__ template<> inline int saturate_cast<int>(softfloat a) {
//	return cvRound(a);
//}
//__HOST_DEVICE__ template<> inline int saturate_cast<int>(softdouble a) {
//	return cvRound(a);
//}
//
//__HOST_DEVICE__ template<> inline int64_t saturate_cast<int64_t>(softfloat a) {
//	return cvRound(a);
//}
//__HOST_DEVICE__ template<> inline int64_t saturate_cast<int64_t>(softdouble a) {
//	return cvRound64(a);
//}
//
///** @brief Saturate cast to unsigned integer and unsigned long long integer
// We intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
// */
//__HOST_DEVICE__ template<> inline unsigned saturate_cast<unsigned>(
//		softfloat a) {
//	return cvRound(a);
//}
//__HOST_DEVICE__ template<> inline unsigned saturate_cast<unsigned>(
//		softdouble a) {
//	return cvRound(a);
//}
//
//__HOST_DEVICE__ template<> inline uint64_t saturate_cast<uint64_t>(
//		softfloat a) {
//	return cvRound(a);
//}
//__HOST_DEVICE__ template<> inline uint64_t saturate_cast<uint64_t>(
//		softdouble a) {
//	return cvRound64(a);
//}

/** @brief Min and Max functions */
__HOST_DEVICE__ inline softfloat min(const softfloat &a, const softfloat &b) {
	return (a > b) ? b : a;
}
__HOST_DEVICE__ inline softdouble min(const softdouble &a,
		const softdouble &b) {
	return (a > b) ? b : a;
}

__HOST_DEVICE__ inline softfloat max(const softfloat &a, const softfloat &b) {
	return (a > b) ? a : b;
}
__HOST_DEVICE__ inline softdouble max(const softdouble &a,
		const softdouble &b) {
	return (a > b) ? a : b;
}

/** @brief Absolute value */
__HOST_DEVICE__ inline softfloat abs(softfloat a) {
	softfloat x;
	x.v = a.v & ((1U << 31) - 1);
	return x;
}
__HOST_DEVICE__ inline softdouble abs(softdouble a) {
	softdouble x;
	x.v = a.v & ((1ULL << 63) - 1);
	return x;
}

/** @brief Exponent

 Special cases:
 - exp(NaN) is NaN
 - exp(-Inf) == 0
 - exp(+Inf) == +Inf
 */
__HOST_DEVICE__ softfloat exp(const softfloat &a);
__HOST_DEVICE__ softdouble exp(const softdouble &a);

/** @brief Natural logarithm

 Special cases:
 - log(NaN), log(x < 0) are NaN
 - log(0) == -Inf
 */
__HOST_DEVICE__ softfloat log(const softfloat &a);
__HOST_DEVICE__ softdouble log(const softdouble &a);

/** @brief Raising to the power

 Special cases:
 - x**NaN is NaN for any x
 - ( |x| == 1 )**Inf is NaN
 - ( |x|  > 1 )**+Inf or ( |x| < 1 )**-Inf is +Inf
 - ( |x|  > 1 )**-Inf or ( |x| < 1 )**+Inf is 0
 - x ** 0 == 1 for any x
 - x ** 1 == 1 for any x
 - NaN ** y is NaN for any other y
 - Inf**(y < 0) == 0
 - Inf ** y is +Inf for any other y
 - (x < 0)**y is NaN for any other y if x can't be correctly rounded to integer
 - 0 ** 0 == 1
 - 0 ** (y < 0) is +Inf
 - 0 ** (y > 0) is 0
 */
__HOST_DEVICE__ softfloat pow(const softfloat &a, const softfloat &b);
__HOST_DEVICE__ softdouble pow(const softdouble &a, const softdouble &b);

/** @brief Cube root

 Special cases:
 - cbrt(NaN) is NaN
 - cbrt(+/-Inf) is +/-Inf
 */
__HOST_DEVICE__ softfloat cbrt(const softfloat &a);

/** @brief Sine

 Special cases:
 - sin(Inf) or sin(NaN) is NaN
 - sin(x) == x when sin(x) is close to zero
 */
__HOST_DEVICE__ softdouble sin(const softdouble &a);

/** @brief Cosine
 *
 Special cases:
 - cos(Inf) or cos(NaN) is NaN
 - cos(x) == +/- 1 when cos(x) is close to +/- 1
 */
__HOST_DEVICE__ softdouble cos(const softdouble &a);

//! @} core_utils_softfloat

}// cv::

#undef __HOST_DEVICE__

#endif
