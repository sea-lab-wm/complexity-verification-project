#include <stdio.h>

void method72() {
   int V1 = 1, V2 = 2;
   int M1, M2; // Added to allow compilation: Use actual variables for conditional values
   // #define M1 2
   // #define M2 1
   // added above lines to allow compilation
   M1 = 2;
   M2 = 1;


   printf("%d %d\n", M1, M2);
}

int main() { method72(); return 0; }
