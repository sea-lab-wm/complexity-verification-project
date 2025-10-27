#include <stdio.h>

void method124() {
   int V1 = 0;
   int V2 = 5;

   if (V1) {
      V2 = 2;
   }
   V2 = V2 * 2;

   printf("%d\n", V2);
}

int main() { method124(); return 0; }
