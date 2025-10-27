#include <stdio.h>

void method64() {
   int V1 = 3;
  
   V1 *= 2;

   int V2 = (V1 += 1);

   printf("%d %d\n", V1, V2);
}

int main() { method64(); return 0; }
