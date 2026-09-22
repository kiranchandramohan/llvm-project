// Link with -Wl,--wrap=malloc,--wrap=free and a scalar-lowered or SME kernel.
#include <cstddef>
#include <cstdio>
#include <initializer_list>

void matmul(bool, bool, int, int, int, float, float *, int, float *, int, float,
           float *, int, bool);
extern "C" void *__real_malloc(std::size_t);
extern "C" void __real_free(void *);
static bool failAllocation;
static unsigned allocations, releases;
static std::size_t requested;
extern "C" void *__wrap_malloc(std::size_t size) {
  ++allocations;
  requested = size;
  return failAllocation ? nullptr : __real_malloc(size);
}
extern "C" void __wrap_free(void *ptr) {
  if (ptr)
    ++releases;
  __real_free(ptr);
}
int main() {
  for (bool fail : {false, true}) {
    float a[] = {2, 3}, b[] = {4, 5}, c = -1;
    failAllocation = fail;
    allocations = releases = 0;
    matmul(false, false, 1, 1, 2, 1, a, 2, b, 1, 0, &c, 1, false);
    if (c != 23 || allocations != 1 || releases != unsigned(!fail) ||
        requested != 20)
      return 1;
    allocations = releases = 0;
    matmul(false, false, 0, 1, 2, 1, nullptr, 2, nullptr, 1, 0, nullptr, 1,
          false);
    if (allocations || releases)
      return 2;
    // K == 0 still applies the epilogue without touching null inputs.
    c = 4;
    matmul(false, false, 1, 1, 0, 1, nullptr, 2, nullptr, 1, 2, &c, 1, false);
    if (c != 8 || allocations != 1 || releases != unsigned(!fail) ||
        requested != 4)
      return 3;
  }
  std::puts("allocation success, failure fallback, empty output and K=0 passed");
}
