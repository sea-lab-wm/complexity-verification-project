#include <stdio.h>

void method15() { 
   int V1 = 0;
   if (V1++ == 0) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}

int main() { method15(); return 0; }
