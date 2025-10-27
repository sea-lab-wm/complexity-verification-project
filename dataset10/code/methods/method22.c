#include <stdio.h>

void method22() { 
   int V1 = 2;
   V1--;
   if (V1 == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}

int main() { method22(); return 0; }
