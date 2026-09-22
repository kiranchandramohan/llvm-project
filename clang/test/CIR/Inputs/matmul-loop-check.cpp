// Link with matmul-loop.cpp compiled normally with -DREFERENCE and with its
// raised linalg payload lowered to scalar LLVM (omit cir-matmul-to-sme).
// Use -Wl,--wrap=malloc,--wrap=free to exercise allocation failure.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define DECLARE(SUFFIX)                                                        \
  extern "C" {                                                                 \
  int reordered##SUFFIX(float *, int, const float *, int, const float *, int,  \
                        int, int, int, float, bool, float, bool, long, int *); \
  int plain##SUFFIX(float *, const float *, const float *, int, int, int,      \
                    int);                                                      \
  void fixed##SUFFIX(float *, const float *, const float *);                   \
  void unconditional##SUFFIX(float *, const float *, const float *, int, int,  \
                             int, float, float);                               \
  void scaled_output##SUFFIX(float *, const float *, const float *, int, int,  \
                             int, float);                                      \
  void multiple##SUFFIX(float *, const float *, const float *, int, int, int); \
  }
DECLARE()
DECLARE(_reference)

static bool failAllocation;
static unsigned allocations;
extern "C" void *__real_malloc(size_t);
extern "C" void __real_free(void *);
extern "C" void *__wrap_malloc(size_t size) {
  ++allocations;
  return failAllocation ? nullptr : __real_malloc(size);
}
extern "C" void __wrap_free(void *pointer) { __real_free(pointer); }

struct Matrices {
  float a[128], b[128], c[128];
  Matrices() {
    for (unsigned i = 0; i != 128; ++i) {
      a[i] = (int(i % 13) - 6) / 7.0f;
      b[i] = (int(i % 11) - 5) / 9.0f;
      c[i] = (int(i % 7) - 3) / 5.0f;
    }
  }
  float *output(unsigned overlap) {
    return overlap == 1 ? a : overlap == 2 ? b : overlap == 3 ? a + 1 : c;
  }
};

static bool equal(const Matrices &actual, const Matrices &expected) {
  auto close = [](float a, float b) {
    return a == b || std::abs(a - b) <= 2e-5f * (1 + std::abs(b)) ||
           (std::isnan(a) && std::isnan(b));
  };
  for (unsigned i = 0; i != 128; ++i)
    if (!close(actual.a[i], expected.a[i]) ||
        !close(actual.b[i], expected.b[i]) ||
        !close(actual.c[i], expected.c[i]))
      return false;
  return true;
}

int main() {
  unsigned cases = 0;
  for (bool fail : {false, true}) {
    failAllocation = fail;
    for (bool ta : {false, true})
      for (bool tb : {false, true})
        for (unsigned overlap : {0u, 1u, 2u, 3u})
          for (int m : {0, 1, 3})
            for (int n : {0, 1, 5})
              for (int k : {-1, 0, 1, 4})
                for (float alpha : {0.0f, 1.0f, -1.25f})
                  for (float beta : {0.0f, -0.0f, 1.0f, -0.5f}) {
                    Matrices actual, expected = actual;
                    int observed = 5, expectedObserved = 5;
                    int result = reordered(actual.output(overlap), m, actual.b,
                                           n, actual.a, k, 8, 8, 8, beta, tb,
                                           alpha, ta, 99, &observed);
                    int expectedResult = reordered_reference(
                        expected.output(overlap), m, expected.b, n, expected.a,
                        k, 8, 8, 8, beta, tb, alpha, ta, 99, &expectedObserved);
                    if (result != expectedResult || observed != 23 ||
                        observed != expectedObserved ||
                        !equal(actual, expected))
                      return std::puts("reordered loop or surrounding effects "
                                       "failed"),
                             1;
                    ++cases;
                  }
    for (unsigned overlap : {0u, 1u, 2u, 3u}) {
      Matrices actual, expected = actual;
      fixed(actual.output(overlap), actual.a, actual.b);
      fixed_reference(expected.output(overlap), expected.a, expected.b);
      if (!equal(actual, expected))
        return std::puts("constant dimensions/layout failed"), 1;
      for (int m : {-2, -1, 0, 2})
        for (int n : {0, 1, 5})
          for (int k : {-1, 0, 1, 4}) {
            actual = Matrices();
            expected = actual;
            int result =
                plain(actual.output(overlap), actual.a, actual.b, m, n, k, 17);
            int expectedResult = plain_reference(
                expected.output(overlap), expected.a, expected.b, m, n, k, 17);
            if (result != 20 || result != expectedResult ||
                !equal(actual, expected))
              return std::puts("plain loop failed"), 1;
            actual = Matrices();
            expected = actual;
            multiple(actual.output(overlap), actual.a, actual.b, m, n, k);
            multiple_reference(expected.output(overlap), expected.a, expected.b,
                               m, n, k);
            if (!equal(actual, expected))
              return std::puts("multiple loops failed"), 1;
            cases += 2;
          }
    }
    Matrices actual, expected;
    std::fill_n(actual.c, 128, std::numeric_limits<float>::quiet_NaN());
    expected = actual;
    unconditional(actual.c, actual.a, actual.b, 3, 5, 4, 1.0f, 0.0f);
    unconditional_reference(expected.c, expected.a, expected.b, 3, 5, 4, 1.0f,
                            0.0f);
    if (!equal(actual, expected) || !std::isnan(actual.c[0]))
      return std::puts("unconditional output read lost"), 1;
    actual = Matrices();
    expected = actual;
    scaled_output(actual.c, actual.a, actual.b, 3, 5, 4, -0.75f);
    scaled_output_reference(expected.c, expected.a, expected.b, 3, 5, 4,
                            -0.75f);
    if (!equal(actual, expected))
      return std::puts("output scale without alpha failed"), 1;
  }
  if (!allocations)
    return std::puts("optimized path was never attempted"), 1;
  std::printf("PASS: %u loop cases, surrounding effects, NaN output reads, and "
              "allocation failure\n",
              cases);
}
