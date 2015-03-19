#include <boost/test/included/unit_test_framework.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "gradient_projection.h"

#define CLOSE_TOL 0.1
#define LOOSE_TOL 10

using boost::unit_test_framework::test_suite;
using namespace GradientProjection;

void test_projection() {
  gsl::matrix n(4, 3);
  gsl::matrix p(1, 1);
  n(0, 0) = 2; n(0, 1) = 1; n(0, 2) = 0;
  n(1, 0) = 1; n(1, 1) = 1; n(1, 2) = 0;
  n(2, 0) = 1; n(2, 1) = 2; n(2, 2) = 0;
  n(3, 0) = 4; n(3, 1) = 1; n(3, 2) = 1;
  gsl::vector g(3);
  g[0] = 0.0; g[1] = -0.1; g[2] = 0.0;
  gsl::vector grad(4);
  grad[0] = 2.0; grad[1] = 4.0; grad[2] = 2.0; grad[3] = -3.0;
  gsl::vector direction;
  gsl::vector correction;

  createProjection(n, g, grad, p, direction, correction);
  BOOST_CHECK_CLOSE(p(0, 0),  1.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(0, 1), -3.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(0, 2),  1.0/11.0, LOOSE_TOL);
  BOOST_CHECK(abs(p(0, 3)) < 1e-10);
  BOOST_CHECK_CLOSE(p(1, 0), -3.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(1, 1),  9.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(1, 2), -3.0/11.0, LOOSE_TOL);
  BOOST_CHECK(abs(p(1, 3)) < 1e-10);
  BOOST_CHECK_CLOSE(p(2, 0),  1.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(2, 1), -3.0/11.0, LOOSE_TOL);
  BOOST_CHECK_CLOSE(p(2, 2),  1.0/11.0, LOOSE_TOL);
  BOOST_CHECK(abs(p(2, 3)) < 1e-10);
  BOOST_CHECK(abs(p(3, 0)) < 1e-10);
  BOOST_CHECK(abs(p(3, 1)) < 1e-10);
  BOOST_CHECK(abs(p(3, 2)) < 1e-10);
  BOOST_CHECK(abs(p(3, 3)) < 1e-10);

  BOOST_CHECK_CLOSE(correction[0], -4.0/110.0, CLOSE_TOL);
  BOOST_CHECK_CLOSE(correction[1],  1.0/110.0, CLOSE_TOL);
  BOOST_CHECK_CLOSE(correction[2],  7.0/110.0, CLOSE_TOL);
  BOOST_CHECK(abs(correction[3]) < 1e-10);

  BOOST_CHECK_CLOSE(direction[0],    8.0/11.0, CLOSE_TOL);
  BOOST_CHECK_CLOSE(direction[1],  -24.0/11.0, CLOSE_TOL);
  BOOST_CHECK_CLOSE(direction[2],    8.0/11.0, CLOSE_TOL);
  BOOST_CHECK(abs(direction[3]) < 1e-10);

  gsl::vector x(4);
  x[0] = 2; x[1] = 2; x[2] = 1; x[3] = 0;

  descend(x, direction, 0.1, 5, correction, grad);
  BOOST_CHECK_CLOSE(x[0], 2.026, CLOSE_TOL);
  BOOST_CHECK_CLOSE(x[1], 1.822, CLOSE_TOL);
  BOOST_CHECK_CLOSE(x[2], 1.126, CLOSE_TOL);
  BOOST_CHECK(abs(x[3]) < 1e-10);
}

void test_constraint_matrix() {
  gsl::matrix n;
  gsl::vector g;
  gsl::vector x(3);


  x[0] = 0.15; x[1] = 0.5; x[2] = 0.2;

  BOOST_CHECK_EQUAL(createActiveConstraints(x, n, g), false);

  x[0] = 0.0;
  BOOST_CHECK_EQUAL(createActiveConstraints(x, n, g), true);
  BOOST_CHECK_EQUAL(n.size1(), x.size());
  // Only one constraint is broken.
  BOOST_CHECK_EQUAL(n.size2(), 1);
  BOOST_CHECK_EQUAL(n(0, 0), 1);
  BOOST_CHECK_EQUAL(n(1, 0), 0);
  BOOST_CHECK_EQUAL(n(2, 0), 0);
  BOOST_CHECK_EQUAL(g.size(), 1);
  BOOST_CHECK_EQUAL(g[0], -SAFETY_BOX);

  x[2] = 0.6;
  BOOST_CHECK_EQUAL(createActiveConstraints(x, n, g), true);
  BOOST_CHECK_EQUAL(n.size1(), x.size());
  // Two constraints are broken.
  BOOST_CHECK_EQUAL(n.size2(), 2);
  BOOST_CHECK_EQUAL(n(0, 0), -1);
  BOOST_CHECK_EQUAL(n(1, 0), -1);
  BOOST_CHECK_EQUAL(n(2, 0), -1);
  BOOST_CHECK_EQUAL(n(0, 1), 1);
  BOOST_CHECK_EQUAL(n(1, 1), 0);
  BOOST_CHECK_EQUAL(n(2, 1), 0);
  BOOST_CHECK_EQUAL(g.size(), 2);
  BOOST_CHECK_CLOSE(g[0], -0.2, CLOSE_TOL);
  BOOST_CHECK_EQUAL(g[1], -SAFETY_BOX);
  
}

test_suite* init_unit_test_suite(int, char* []) {
  test_suite* test= BOOST_TEST_SUITE( "Testing Gradient Projection" );
  test->add( BOOST_TEST_CASE( &test_constraint_matrix ),    0);
  test->add( BOOST_TEST_CASE( &test_projection        ),    0);
  return test;
}
