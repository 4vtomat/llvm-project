#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

extern int __start___llvm_loop_prof;
extern int __stop___llvm_loop_prof;

// Default profile output name
static const char fileName[] = "./loopProfile.output";

struct anchor {
  uint64_t *counters;         // counters are 64bits
  uint64_t *exiting_counters; // counters are 64bits
  uint64_t *trip_counters;
  uint64_t *min_trip_counters;
  uint64_t *max_trip_counters;
  char **loopNames;
  size_t size;
  char *executed_flags;
};

void dump_loop_profile() {
  struct anchor *an = (struct anchor *)&__start___llvm_loop_prof;
  struct anchor *end = (struct anchor *)&__stop___llvm_loop_prof;
  FILE *fout = NULL;

  if (getenv("LOOP_PROFILE") != NULL) {
    fout = fopen(getenv("LOOP_PROFILE"), "w");
  } else {
    fout = fopen(fileName, "w");
  }

  if (!fout) {
    fprintf(stderr, "[LoopProfiler] Cannot open file!\n");
    return;
  }

  fprintf(fout, "Name, exit_count, trip_count, trip_min, trip_max\n");
  while (an < end) {
    size_t size = (size_t)an->size;

    for (size_t i = 0; i < size; ++i) {
      if (!an->counters || !an->loopNames || !an->trip_counters ||
          !an->trip_counters || !an->min_trip_counters ||
          !an->max_trip_counters)
        break;
      fprintf(fout, "%s, %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "\n",
              an->loopNames[i], an->counters[i], an->trip_counters[i],
              an->min_trip_counters[i], an->max_trip_counters[i]);
    }

    /* Linker will align the struct to 16 bytes */
    an = (struct anchor *)((char *)an + ((sizeof(struct anchor) + 15) & -16));
  }

  fclose(fout);
  return;
}

extern int __loop_prof_info_collection(struct anchor *Anchor, int iter,
                                       uint64_t trip_count);

// Rutime function which helping calculate the min/max trip count in runtime
int __loop_prof_info_collection(struct anchor *Anchor, int i,
                                uint64_t trip_count) {
  uint64_t min = Anchor->min_trip_counters[i];
  uint64_t max = Anchor->max_trip_counters[i];
  if (trip_count == 0)
    return 0;

  Anchor->counters[i] += 1;
  Anchor->min_trip_counters[i] = (trip_count < min) ? trip_count : min;
  Anchor->max_trip_counters[i] = (trip_count > max) ? trip_count : max;
  Anchor->trip_counters[i] += trip_count;
  return 0;
}
