#include <stdio.h>

void method89() {
   int V1;
   for (int V2 = 0; V2 < 2; V2++) {
      V1 = (V2 < 1);
      if (V1) {
         V1 = V2 + 5;
      } else {
         V1 = V1 + 2;
      }
   }
   printf("%d\n", V1);
}

int main() { method89(); return 0; }
