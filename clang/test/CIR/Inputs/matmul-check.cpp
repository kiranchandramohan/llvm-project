#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

void matmul(bool, bool, int, int, int, float, float *, int, float *, int, float,
           float *, int, bool);
void matmul_reference(bool, bool, int, int, int, float, float *, int, float *,
                     int, float, float *, int, bool);

int main() {
  unsigned cases = 0;
  for (bool ta : {false, true})
    for (bool tb : {false, true})
      for (int m : {0, 1, 3, 7, 17})
        for (int n : {0, 1, 5, 9, 19})
          for (int k : {-2, 0, 1, 6, 13})
            for (float alpha : {0.0f, 1.0f, -1.25f})
              for (float beta : {0.0f, -0.0f, 1.0f, -0.5f}) {
                int kp = std::max(k, 0);
                int lda = (ta ? m : kp) + 3;
                int ldb = (tb ? kp : n) + 2;
                int ldc = n + 5;
                std::vector<float> a((ta ? kp : m) * lda + 1);
                std::vector<float> b((tb ? n : kp) * ldb + 1);
                std::vector<float> c(m * ldc + 1, -987.0f);
                for (unsigned i = 0; i < a.size(); ++i)
                  a[i] = (int(i % 19) - 9) / 7.0f;
                for (unsigned i = 0; i < b.size(); ++i)
                  b[i] = (int(i % 13) - 6) / 11.0f;
                for (int i = 0; i < m; ++i)
                  for (int j = 0; j < n; ++j)
                    c[i * ldc + j] =
                        beta == 0 ? std::numeric_limits<float>::quiet_NaN()
                                  : (i - j) / 3.0f;
                auto expected = c;
                matmul_reference(ta, tb, m, n, k, alpha, a.data(), lda, b.data(),
                                ldb, beta, expected.data(), ldc, false);
                matmul(ta, tb, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                      beta, c.data(), ldc, true);
                for (unsigned i = 0; i < c.size(); ++i) {
                  if (!(std::abs(c[i] - expected[i]) <=
                            2e-5f * (1 + std::abs(expected[i])) ||
                        (std::isnan(c[i]) && std::isnan(expected[i])))) {
                    std::printf("FAIL ta=%d tb=%d m=%d n=%d k=%d alpha=%g "
                                "beta=%g at %u: %g != %g\n",
                                ta, tb, m, n, k, alpha, beta, i, c[i],
                                expected[i]);
                    return 1;
                  }
                }
                ++cases;
              }
  // Signed/zero strides, overlapping rows of C, and A/B aliasing. The
  // contract only excludes overlap between the output and the inputs.
  unsigned stridedCases = 0;
  for (bool ta : {false, true})
    for (bool tb : {false, true})
      for (int lda : {-11, 0, 11})
        for (int ldb : {-13, 0, 13})
          for (int ldc : {-9, 0, 9})
            for (bool aliasInputs : {false, true}) {
              std::vector<float> a(512), b(512), c(512);
              for (unsigned i = 0; i < a.size(); ++i) {
                a[i] = (int(i % 17) - 8) / 7.0f;
                b[i] = (int(i % 11) - 5) / 9.0f;
                c[i] = (int(i % 7) - 3) / 5.0f;
              }
              auto expected = c;
              float *ap = a.data() + 256;
              float *bp = (aliasInputs ? a.data() : b.data()) + 256;
              matmul_reference(ta, tb, 3, 5, 7, 1.25f, ap, lda, bp, ldb, -0.5f,
                              expected.data() + 256, ldc, false);
              matmul(ta, tb, 3, 5, 7, 1.25f, ap, lda, bp, ldb, -0.5f,
                    c.data() + 256, ldc, false);
              for (unsigned i = 0; i < c.size(); ++i)
                if (!(std::abs(c[i] - expected[i]) <=
                      2e-5f * (1 + std::abs(expected[i])))) {
                  std::printf("FAIL signed strides/aliasing at %u\n", i);
                  return 1;
                }
              ++stridedCases;
            }
  // Zero K still evaluates alpha * zero, including infinity producing NaN.
  float exceptional = 2.0f;
  matmul(false, false, 1, 1, 0, std::numeric_limits<float>::infinity(), nullptr,
        1, nullptr, 1, 0, &exceptional, 1, false);
  if (!std::isnan(exceptional))
    return 1;
  // A NaN beta takes the branch that reads C.
  exceptional = 2.0f;
  matmul(false, false, 1, 1, 0, 1, nullptr, 1, nullptr, 1,
        std::numeric_limits<float>::quiet_NaN(), &exceptional, 1, false);
  if (!std::isnan(exceptional))
    return 1;
  // Empty output must not access any input, even for positive width.
  matmul(false, false, -1, 10, 5, 1, nullptr, 7, nullptr, 12, 0, nullptr, 12,
        false);
  matmul(true, true, 10, 0, 5, 1, nullptr, 12, nullptr, 7, 1, nullptr, 1, false);
  std::printf("PASS: %u differential cases, %u signed-stride/alias cases, "
              "and exceptional/empty-output cases\n",
              cases, stridedCases);
}
