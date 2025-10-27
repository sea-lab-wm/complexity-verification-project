#include <stdio.h>

void method113() {
   int V1 = 3;

   for (int V2 = 0; V2 < 3; V2++) V1++; V1++;

   printf("%d\n", V1);
}

int main() { method113(); return 0; }
