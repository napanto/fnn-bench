// Spinners for the SMT sibling test: "pause" touches only registers, "l1" walks a 64 KiB buffer.
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "l1")) {
        static volatile char b[65536];
        for (;;)
            for (int i = 0; i < 65536; i += 64) b[i]++;
    }
    for (;;) __asm__ volatile("pause");
}
