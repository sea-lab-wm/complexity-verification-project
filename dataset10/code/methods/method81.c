#include <stdio.h>

void method81() {
   int V1 = 1;
   int V2 = 5;

   V1 == V2 && ++V1 || ++V2;

   printf("%d %d\n", V1, V2);
}

int main() { method81(); return 0; }
