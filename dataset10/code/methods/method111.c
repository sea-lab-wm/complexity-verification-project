#include <stdio.h>

void method111() {
   int V1 = 4;

   int V2 = 0;
   while (V2 < 3) V2++; V1++;

   printf("%d %d\n", V1, V2);
}

int main() { method111(); return 0; }
