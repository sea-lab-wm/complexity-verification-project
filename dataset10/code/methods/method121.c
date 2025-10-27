#include <stdio.h>

void method121() {
   int V1 = 0;
   int V2 = 2;

   if (V1) {}
      V2 = 4;

   printf("%d\n", V2);
}

int main() { method121(); return 0; }
