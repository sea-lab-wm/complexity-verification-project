#include <stdio.h>

void method45() {
   int V1[] = {4, 2, 7, 5};
   int *V2 = V1 + 1;
   printf("%d\n", *V2);
}

int main() { method45(); return 0; }
