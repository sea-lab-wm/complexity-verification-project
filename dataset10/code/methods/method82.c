#include <stdio.h>

void method82() {
   int V1 = 2;
   int V2 = 6;

   if (V1 == V2) {
      ++V1;
   } else {
      ++V2;
   }

   printf("%d %d\n", V1, V2);
}

int main() { method82(); return 0; }
