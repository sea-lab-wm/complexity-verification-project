#include <stdio.h>

void method52() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4;
   if (V1 == 2) {
      if (V3 == 2) {
         V4 = 1;
      } else {
         V4 = 2;
      }
   } else {
      if (V2 == 2) {
         V4 = 3;
      } else {
         V4 = 4;
      }
   }

   printf("%d\n", V4);
}

int main() { method52(); return 0; }
