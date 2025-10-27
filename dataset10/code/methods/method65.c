#include <stdio.h>

void method65() {
   int V1 = 3;
   int V2 = (V1 += 1, V1 *= 2);

   printf("%d %d\n", V1, V2);
}

int main() { method65(); return 0; }
