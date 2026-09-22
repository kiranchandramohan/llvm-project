// Differential execution of the runtime alias dispatcher. Build this harness
// against a renamed C++ reference and either scalar-lowered or SME MatMul.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

void matmul(bool, bool, int, int, int, float, float *, int, float *, int, float,
           float *, int, bool);
void matmul_reference(bool, bool, int, int, int, float, float *, int, float *,
                     int, float, float *, int, bool);

int main() {
  unsigned cases = 0, exactAliases = 0;
  for (bool ta : {false, true})
    for (bool tb : {false, true})
      for (int lda : {-11, 0, 11})
        for (int ldb : {-13, 0, 13})
          for (int ldc : {-9, 0, 9})
            for (int bo : {324, 600})
              for (int co : {300, 320, 321, 324, 326, 560, 600, 602, 900})
                for (float beta : {0.0f, -0.5f}) {
                  std::vector<float> actual(1024);
                  for (unsigned i = 0; i < actual.size(); ++i)
                    actual[i] = (int((i * 17) % 31) - 15) / 23.0f;
                  auto expected = actual;
                  matmul_reference(ta, tb, 3, 5, 7, 0.75f, expected.data() + 320,
                                  lda, expected.data() + bo, ldb, beta,
                                  expected.data() + co, ldc, false);
                  matmul(ta, tb, 3, 5, 7, 0.75f, actual.data() + 320, lda,
                        actual.data() + bo, ldb, beta, actual.data() + co, ldc,
                        false);
                  for (unsigned i = 0; i < actual.size(); ++i)
                    if (!(actual[i] == expected[i] ||
                          (std::isnan(actual[i]) && std::isnan(expected[i])) ||
                          std::abs(actual[i] - expected[i]) <=
                              5e-5f * (1 + std::abs(expected[i])))) {
                      std::printf("FAIL ta=%d tb=%d lda=%d ldb=%d ldc=%d "
                                  "B=%d C=%d beta=%g at %u: %g != %g\n",
                                  ta, tb, lda, ldb, ldc, bo, co, beta, i,
                                  actual[i], expected[i]);
                      return 1;
                    }
                  // Exact base aliases must take the unmodified fallback.
                  // This comparison also detects accidental FMA insertion in
                  // a fallback compiled with -ffp-contract=off.
                  if (co == 320 || co == bo) {
                    if (std::memcmp(actual.data(), expected.data(),
                                    actual.size() * sizeof(float))) {
                      std::puts("FAIL: alias fallback changed FP semantics");
                      return 1;
                    }
                    ++exactAliases;
                  }
                  ++cases;
                }
  std::printf(
      "PASS: %u overlap/disjoint cases, including %u bitwise fallback checks\n",
      cases, exactAliases);
}
