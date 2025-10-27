#include <stdio.h>

void method66() {
   int V1 = 3;
  
   V1 += 1;

   int V2 = (V1 *= 2);

   printf("%d %d\n", V1, V2);
}

int main() { method66(); return 0; }
