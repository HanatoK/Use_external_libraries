#include <array>
#include <cmath>
#include <cuda/std/array>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#define HOST_DEVICE __device__ __host__
#else
#define HOST_DEVICE
#endif

constexpr int max_blocks = 64;

struct rvector {
  double x;
  double y;
  double z;
};

struct rmatrix {
  double xx, xy, xz, yx, yy, yz, zx, zy, zz;
  inline friend rvector operator * (rmatrix const &m,
                                    rvector const &r)
  {
    return rvector{m.xx*r.x + m.xy*r.y + m.xz*r.z,
                   m.yx*r.x + m.yy*r.y + m.yz*r.z,
                   m.zx*r.x + m.zy*r.y + m.zz*r.z};
  }
};

struct quaternion {
  double q0;
  double q1;
  double q2;
  double q3;
  inline void set_from_euler_angles(double phi_in,
                                    double theta_in,
                                    double psi_in)
  {
    q0 = ( (std::cos(phi_in/2.0)) * (std::cos(theta_in/2.0)) * (std::cos(psi_in/2.0)) +
           (std::sin(phi_in/2.0)) * (std::sin(theta_in/2.0)) * (std::sin(psi_in/2.0)) );

    q1 = ( (std::sin(phi_in/2.0)) * (std::cos(theta_in/2.0)) * (std::cos(psi_in/2.0)) -
           (std::cos(phi_in/2.0)) * (std::sin(theta_in/2.0)) * (std::sin(psi_in/2.0)) );

    q2 = ( (std::cos(phi_in/2.0)) * (std::sin(theta_in/2.0)) * (std::cos(psi_in/2.0)) +
           (std::sin(phi_in/2.0)) * (std::cos(theta_in/2.0)) * (std::sin(psi_in/2.0)) );

    q3 = ( (std::cos(phi_in/2.0)) * (std::cos(theta_in/2.0)) * (std::sin(psi_in/2.0)) -
           (std::sin(phi_in/2.0)) * (std::sin(theta_in/2.0)) * (std::cos(psi_in/2.0)) );
  }
  inline rvector rotate(rvector const &v) const
  {
    return ( (*this) * quaternion{0.0, v.x, v.y, v.z} *
             this->conjugate() ).get_vector();
  }
  inline quaternion conjugate() const
  {
    return quaternion{q0, -q1, -q2, -q3};
  }
  friend inline quaternion operator * (quaternion const &h,
                                       quaternion const &q)
  {
    return quaternion{h.q0*q.q0 - h.q1*q.q1 - h.q2*q.q2 - h.q3*q.q3,
                      h.q0*q.q1 + h.q1*q.q0 + h.q2*q.q3 - h.q3*q.q2,
                      h.q0*q.q2 + h.q2*q.q0 + h.q3*q.q1 - h.q1*q.q3,
                      h.q0*q.q3 + h.q3*q.q0 + h.q1*q.q2 - h.q2*q.q1};
  }
  inline rmatrix rotation_matrix() const
  {
    rmatrix R;

    R.xx = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    R.yy = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    R.zz = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    R.xy = 2.0 * (q1*q2 - q0*q3);
    R.xz = 2.0 * (q0*q2 + q1*q3);

    R.yx = 2.0 * (q0*q3 + q1*q2);
    R.yz = 2.0 * (q2*q3 - q0*q1);

    R.zx = 2.0 * (q1*q3 - q0*q2);
    R.zy = 2.0 * (q0*q1 + q2*q3);

    return R;
  }
  inline rvector get_vector() const
  {
    return rvector{q1, q2, q3};
  }

  inline HOST_DEVICE quaternion position_derivative_inner(rvector const &pos,
                                            rvector const &vec) const {
    return quaternion{2.0 * (vec.x * ( q0 * pos.x - q3 * pos.y + q2 * pos.z) +
                             vec.y * ( q3 * pos.x + q0 * pos.y - q1 * pos.z) +
                             vec.z * (-q2 * pos.x + q1 * pos.y + q0 * pos.z)),
                      2.0 * (vec.x * ( q1 * pos.x + q2 * pos.y + q3 * pos.z) +
                             vec.y * ( q2 * pos.x - q1 * pos.y - q0 * pos.z) +
                             vec.z * ( q3 * pos.x + q0 * pos.y - q1 * pos.z)),
                      2.0 * (vec.x * (-q2 * pos.x + q1 * pos.y + q0 * pos.z) +
                             vec.y * ( q1 * pos.x + q2 * pos.y + q3 * pos.z) +
                             vec.z * (-q0 * pos.x + q3 * pos.y - q2 * pos.z)),
                      2.0 * (vec.x * (-q3 * pos.x - q0 * pos.y + q1 * pos.z) +
                             vec.y * ( q0 * pos.x - q3 * pos.y + q2 * pos.z) +
                             vec.z * ( q1 * pos.x + q2 * pos.y + q3 * pos.z))};
  }

  inline HOST_DEVICE cuda::std::array<double, 4> derivative_element_wise_product_sum(const double (&C)[3][3]) const {
    return cuda::std::array<double, 4>{{
      2.0 * ( q0 * C[0][0] - q3 * C[0][1] + q2 * C[0][2] +
              q3 * C[1][0] + q0 * C[1][1] - q1 * C[1][2] +
             -q2 * C[2][0] + q1 * C[2][1] + q0 * C[2][2]),
      2.0 * ( q1 * C[0][0] + q2 * C[0][1] + q3 * C[0][2] +
              q2 * C[1][0] - q1 * C[1][1] - q0 * C[1][2] +
              q3 * C[2][0] + q0 * C[2][1] - q1 * C[2][2]),
      2.0 * (-q2 * C[0][0] + q1 * C[0][1] + q0 * C[0][2] +
              q1 * C[1][0] + q2 * C[1][1] + q3 * C[1][2] +
             -q0 * C[2][0] + q3 * C[2][1] - q2 * C[2][2]),
      2.0 * (-q3 * C[0][0] - q0 * C[0][1] + q1 * C[0][2] +
              q0 * C[1][0] - q3 * C[1][1] + q2 * C[1][2] +
              q1 * C[2][0] + q2 * C[2][1] + q3 * C[2][2])
    }};
  }
};

void project1_cuda(const rvector* pos1, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream);

void project2_cuda(const rvector* pos1, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream);
