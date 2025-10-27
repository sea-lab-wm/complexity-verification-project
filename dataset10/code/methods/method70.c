#include <stdio.h>

void method70() {
   #define M1 1
   #define M2 2

   int V1;
   V1 = 4;

   int V2 = 1 + V1;

   printf("%d %d\n", V1, V2);
}

int main() { method70(); return 0; }
