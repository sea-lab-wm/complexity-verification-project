#include <stdio.h>

void method123() {
   int V1 = 0;
   int V2 = 1;

   if (V1) {
      V2 = 2;
   }
      V2 = V2 * 3;

   printf("%d\n", V2);
}

int main() { method123(); return 0; }
