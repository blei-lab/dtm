#ifndef __MATH_GSL_MATRIX__
#define __MATH_GSL_MATRIX__

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cblas.h>

class GslMatrixItem {
 public:
  GslMatrixItem(gsl_matrix* ptr, size_t index1, size_t index2) :
    ptr_(ptr),
    index1_(index1),
    index2_(index2) { }

  operator const double() {
    return gsl_matrix_get(ptr_, index1_, index2_);
  }

  double operator =(const double v) {
    gsl_matrix_set(ptr_, index1_, index2_, v);
    return v;
  }

  double operator +=(const double v) {
    double old_v = gsl_matrix_get(ptr_, index1_, index2_);
    gsl_matrix_set(ptr_, index1_, index2_, v + old_v);
    return v + old_v;
  }
 private:
  gsl_matrix* ptr_;
  size_t index1_;
  size_t index2_;
};

class GslMatrixBase {
 public:
  GslMatrixBase& operator=(const double v) {
    if (v == 0.0) {
      SetZero();
    } else {
      SetAll(v);
    }
    return *this;
  }

  GslMatrixItem operator()(const size_t index1, const size_t index2) const {
    assert(ptr_ != NULL);
    return GslMatrixItem(ptr_, index1, index2);
  }

  void SetZero() {
    assert(ptr_ != NULL);
    gsl_matrix_set_zero(ptr_);
  }

  void SetAll(const double v) {
    assert(ptr_ != NULL);
    gsl_matrix_set_all(ptr_, v);
  }
  
  void Reset(gsl_matrix* val) {
    if(ptr_ != NULL) {
      gsl_matrix_free(ptr_);
    }
    ptr_ = val;
  }

  int Fprintf(FILE* stream, const char* format) const {
    assert(ptr_ != NULL);
    return gsl_matrix_fprintf(stream, ptr_, format);
  }

  int Fscanf(FILE* stream) {
    assert(ptr_ != NULL);
    return gsl_matrix_fscanf(stream, ptr_);
  }

  void Set(const int i, const int j, double val) {
    gsl_matrix_set(ptr_, i, j, val);
  }

  /*
    double operator()(const int nCol, const int nRow) {
    return gsl_matrix_get(ptr_, nCol, nRow);
    }
  */

  int size1() const {
    return ptr_->size1;
  }

  int size2() const {
    return ptr_->size2;
  }

  double Trace() const {
    double val = 0;
    assert(ptr_ != NULL);
    assert(ptr_->size1 == ptr_->size2);
    for (size_t ii = 0; ii < ptr_->size1; ++ii) {
      val += gsl_matrix_get(ptr_, ii, ii);
    }
    return val;
  }

  double Sum() const {
    double val = 0;
    assert(ptr_ != NULL);
    for (size_t ii = 0; ii < ptr_->size1; ++ii) {
      for (size_t jj = 0; jj < ptr_->size2; ++jj) {
	val += gsl_matrix_get(ptr_, ii, jj);
      }
    }
    return val;
  }

  /*
   * Apply the transpose of this matrix to a vector x and store the result.

   int TransMul(const GslVector& x, GslVector& res, double scale = 0.0) {
   return gsl_blas_dgemv(CblasTrans, 1.0, ptr_, x.ptr(), scale, res.ptr());
   }
   
   int Mul(const GslVector& x, GslVector& res, double scale = 0.0) {
   return gsl_blas_dgemv(CblasNoTrans, 1.0, ptr_, x.ptr(), scale, res.ptr());
   }
  */

  const gsl_matrix* ptr() const { return ptr_; }
  gsl_matrix* mutable_ptr() { return ptr_; }

 protected:
  GslMatrixBase() : ptr_(NULL) {
  }  
  gsl_matrix* ptr_;

 private:
  GslMatrixBase(const GslMatrixBase&) { }
};

class GslMatrix : public GslMatrixBase {
 public:
  GslMatrix(const size_t size1, const size_t size2) : GslMatrixBase() {
    Allocate(size1, size2);
  }

  void Allocate(const size_t size1, const size_t size2) {
    assert(ptr_ == NULL);
    ptr_ = gsl_matrix_alloc(size1, size2);
  }

  GslMatrix() : GslMatrixBase() {
  }

  GslMatrix(gsl_matrix* val) : GslMatrixBase() {
    ptr_ = val;
  }

  ~GslMatrix() {
    if(ptr_ != NULL) {
      gsl_matrix_free(ptr_);
    }
  }

  GslMatrixBase& operator=(const double v) {
    GslMatrixBase::operator=(v);
    return *this;
  }
 private:
  GslMatrix(const GslMatrix&) { }
};


class GslSubmatrix : public GslMatrixBase {
 public:
 GslSubmatrix(GslMatrixBase& matrix, size_t k1, size_t k2, size_t n1, size_t n2) :
  view_(gsl_matrix_submatrix(matrix.mutable_ptr(), k1, k2, n1, n2)) {    
     ptr_ = &view_.matrix;
  }

 GslSubmatrix(gsl_matrix* matrix, size_t k1, size_t k2, size_t n1, size_t n2) :
  view_(gsl_matrix_submatrix(matrix, k1, k2, n1, n2)) {    
     ptr_ = &view_.matrix;
  }

  GslMatrixBase& operator=(const double v) {
    GslMatrixBase::operator=(v);
    return *this;
  }
 private:
  gsl_matrix_view view_;
  GslSubmatrix(const GslSubmatrix&) { }
};

#endif  // __MATH_GSL_MATRIX__
