#include <stdio.h>

void method101() {
   int V1 = 0;

   V1 = V1;

   printf("%d\n", V1);
}

int main() { method101(); return 0; }
