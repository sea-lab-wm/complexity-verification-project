#include <stdio.h>

void method63() {
   int V1 = 3;
   int V2 = (V1 *= 2, V1 += 1);

   printf("%d %d\n", V1, V2);
}

int main() { method63(); return 0; }
