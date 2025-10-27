#include <stdio.h>

void method71() {
   int V1 = 1, V2 = 2;
   int M1, M2; // Added to allow compilation: Use actual variables for conditional values
   
   if (V1 < V2) {
      // Removed below lines to allow compilation
      // #define M1 1
      // #define M2 2

      M1 = 1;
      M2 = 2;

   } else {
      // Removed below lines to allow compilation
      // #define M1 2
      // #define M2 1

      M1 = 2;
      M2 = 1;
   }

   printf("%d %d\n", M1, M2);
}

int main() { method71(); return 0; }
