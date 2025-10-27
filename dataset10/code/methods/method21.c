#include <stdio.h>

void method21() { 
   int V1 = 2;
   if (--V1 == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}

int main() { method21(); return 0; }
