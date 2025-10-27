#include <stdio.h>

void method99() {
   int V1 = 1;

   if (0) {
      V1 = 3;
   }

   printf("%d\n", V1);
}

int main() { method99(); return 0; }
