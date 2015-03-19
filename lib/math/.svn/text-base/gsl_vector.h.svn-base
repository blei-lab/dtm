#ifndef __MATH_GSL_VECTOR__
#define __MATH_GSL_VECTOR__

#include <assert.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_cblas.h>

#include <gsl/gsl_matrix.h>
#include "math/vectorops.h"

class GslVectorItem {
 public:
  GslVectorItem(gsl_vector* ptr, size_t index) :
    ptr_(ptr),
    index_(index) { }

  operator const double() {
    return gsl_vector_get(ptr_, index_);
  }

  double operator =(const double v) {
    gsl_vector_set(ptr_, index_, v);
    return v;
  }

  double operator +=(const double v) {
    double oldv = gsl_vector_get(ptr_, index_);
    gsl_vector_set(ptr_, index_, oldv + v);
    return oldv + v;
  }

  double operator -=(const double v) {
    double oldv = gsl_vector_get(ptr_, index_);
    gsl_vector_set(ptr_, index_, oldv - v);
    return oldv - v;
  }
 private:
  gsl_vector* ptr_;
  size_t index_;
};

class GslVectorBase {
 public:
  /*
  double operator[](const size_t index) const {
    assert(ptr_ != NULL);
    return gsl_vector_get(ptr_, index);
  }
  */

  GslVectorItem operator[](const size_t index) const {
    assert(ptr_ != NULL);
    return GslVectorItem(ptr_, index);
  }

  GslVectorBase& operator+=(const gsl_vector* x) {
    assert(ptr_ != NULL);
    gsl_vector_add(ptr_, x);
    return *this;
  }

  GslVectorBase& operator-=(const gsl_vector* x) {
    assert(ptr_ != NULL);
    gsl_vector_sub(ptr_, x);
    return *this;
  }

  GslVectorBase& operator+=(const GslVectorBase& x) {
    assert(ptr_ != NULL);
    gsl_vector_add(ptr_, x.ptr());
    return *this;
  }

  GslVectorBase& operator-=(const GslVectorBase& x) {
    assert(ptr_ != NULL);
    gsl_vector_sub(ptr_, x.ptr());
    return *this;
  }

  GslVectorBase& operator*=(const gsl_vector* x) {
    assert(ptr_ != NULL);
    gsl_vector_mul(ptr_, x);
    return *this;
  }

  GslVectorBase& operator*=(const GslVectorBase& x) {
    assert(ptr_ != NULL);
    gsl_vector_mul(ptr_, x.ptr());
    return *this;
  }

  GslVectorBase& operator/=(const GslVectorBase& x) {
    assert(ptr_ != NULL);
    gsl_vector_div(ptr_, x.ptr());
    return *this;
  }

  double Sum() const {
    return gsl_blas_dsum(ptr_);
  }

  double L2Norm() const {
    return gsl_blas_dnrm2(ptr_);
  }

  void Normalize() const {
    assert(ptr_ != NULL);
    double s = Sum();
    gsl_vector_scale(ptr_, 1. / s);
  }

  size_t size() const {
    return ptr_->size;
  }

  GslVectorBase& operator*=(double v) {
    assert(ptr_ != NULL);
    gsl_vector_scale(ptr_, v);
    return *this;
  }

  GslVectorBase& operator/=(double v) {
    assert(ptr_ != NULL);
    gsl_vector_scale(ptr_, 1. / v);
    return *this;
  }

  // Note that the standalone product is a dot product!
  const double operator*(const gsl_vector* x) const {
    double result;
    assert(ptr_ != NULL);
    gsl_blas_ddot(ptr_, x, &result);
    return result;
  }

  const double operator*(const GslVectorBase& x) const {
    double result;
    assert(ptr_ != NULL);
    gsl_blas_ddot(ptr_, x.ptr(), &result);
    return result;
  }

  GslVectorBase& operator+=(const double x) {
    for (size_t ii = 0; ii < ptr_->size; ++ii) {
      gsl_vector_set(ptr_, ii, gsl_vector_get(ptr_, ii) + x);
    }
    return *this;
  }

  GslVectorBase& operator-=(const double x) {
    for (size_t ii = 0; ii < ptr_->size; ++ii) {
      gsl_vector_set(ptr_, ii, gsl_vector_get(ptr_, ii) - x);
    }
    return *this;
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    assert(ptr_ != NULL);
    gsl_vector_memcpy(ptr_, x);

    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    assert(ptr_ != NULL);
    return *this = x.ptr();
  }

  GslVectorBase& operator=(const double v) {
    if (v == 0.0) {
      SetZero();
    } else {
      SetAll(v);
    }
    return *this;
  }

  void SetZero() {
    assert(ptr_ != NULL);
    gsl_vector_set_zero(ptr_);
  }

  void SetAll(const double v) {
    assert(ptr_ != NULL);
    gsl_vector_set_all(ptr_, v);
  }

  int Fprintf(FILE* stream, const char* format) const {
    assert(ptr_ != NULL);
    return gsl_vector_fprintf(stream, ptr_, format);
  }

  int Fscanf(FILE* stream) {
    assert(ptr_ != NULL);
    return gsl_vector_fscanf(stream, ptr_);
  }

  const gsl_vector* ptr() const { return ptr_; }
  gsl_vector* mutable_ptr() { return ptr_; }

 protected:
  GslVectorBase() : ptr_(NULL) { }
  gsl_vector* ptr_;

 private:
  GslVectorBase(const GslVectorBase&) { }
};

class GslVector : public GslVectorBase {
 public:
  GslVector(const size_t size) : GslVectorBase() {
    Allocate(size);
  }

  GslVector() : GslVectorBase() {
  }

  ~GslVector() {
    if (ptr_ != NULL) {
      gsl_vector_free(ptr_);
    }
  }

  void Reset(gsl_vector* val) {
    if (ptr_ != NULL) {
      gsl_vector_free(ptr_);
    }
    ptr_ = val;
  }

  void Allocate(const size_t size) {
    assert(ptr_ == NULL);
    ptr_ = gsl_vector_alloc(size);
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const double v) {
    GslVectorBase::operator=(v);
    return *this;
  }
 private:
  GslVector(const GslVector&) { }
};

class GslMatrixRow : public GslVectorBase {
 public:
  GslMatrixRow(GslMatrix& matrix, const size_t row) :
   view_(gsl_matrix_row(matrix.mutable_ptr(), row)) {    
     ptr_ = &view_.vector;
  }

  GslMatrixRow(gsl_matrix* matrix, const size_t row) :
   view_(gsl_matrix_row(matrix, row)) {    
     ptr_ = &view_.vector;
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const double v) {
    GslVectorBase::operator=(v);
    return *this;
  }
 private:
  gsl_vector_view view_;
  GslMatrixRow(const GslMatrixRow&) { }
};

class GslMatrixColumn : public GslVectorBase {
 public:
  GslMatrixColumn(GslMatrix& matrix, const size_t col) :
   view_(gsl_matrix_column(matrix.mutable_ptr(), col)) {    
     ptr_ = &view_.vector;
  }

  GslMatrixColumn(gsl_matrix* matrix, const size_t col) :
   view_(gsl_matrix_column(matrix, col)) {    
     ptr_ = &view_.vector;
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const double v) {
    GslVectorBase::operator=(v);
    return *this;
  }
 private:
  gsl_vector_view view_;
  GslMatrixColumn(const GslMatrixColumn&) { }
};

class GslMatrixDiagonal : public GslVectorBase {
 public:
  GslMatrixDiagonal(GslMatrix& matrix) :
   view_(gsl_matrix_diagonal(matrix.mutable_ptr())) {    
     ptr_ = &view_.vector;
  }

  GslMatrixDiagonal(gsl_matrix* matrix) :
   view_(gsl_matrix_diagonal(matrix)) {    
     ptr_ = &view_.vector;
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const double v) {
    GslVectorBase::operator=(v);
    return *this;
  }
 private:
  gsl_vector_view view_;
  GslMatrixDiagonal(const GslMatrixDiagonal&) { }
};

class GslSubvector : public GslVectorBase {
 public:
 GslSubvector(GslVectorBase& vector, size_t i, size_t n) :
  view_(gsl_vector_subvector(vector.mutable_ptr(), i, n)) {    
     ptr_ = &view_.vector;
  }

  GslVectorBase& operator=(const gsl_vector* x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const GslVectorBase& x) {
    GslVectorBase::operator=(x);
    return *this;
  }

  GslVectorBase& operator=(const double v) {
    GslVectorBase::operator=(v);
    return *this;
  }
 private:
  gsl_vector_view view_;
  GslSubvector(const GslSubvector&) { }
};

#endif  // __MATH_GSL_VECTOR__

