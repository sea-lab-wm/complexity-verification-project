#include <stdio.h>

void method53() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4 = V1 == 3 ? V2 : V3;

   printf("%d\n", V4);
}

int main() { method53(); return 0; }
