#include <stdio.h>

void method17() { 
   int V1 = 2;
   if (V1-- == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}

int main() { method17(); return 0; }
