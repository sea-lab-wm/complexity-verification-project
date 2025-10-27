#include <stdio.h>

void method86() {
   int V1[6];
   int V2 = 5;

   while (V2) {
      V1[5 - V2] = V2;
      V2 = V2 - 1;
   }

   printf("%d %d\n", V1[1], V2);
}

int main() { method86(); return 0; }
