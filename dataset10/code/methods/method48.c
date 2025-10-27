#include <stdio.h>

void method48() {
   int V1[] = {3, 2, 9, 4};
   int *V2 = &V1[1];
   V2 = &V2[2];
   printf("%d\n", *V2);
}

int main() { method48(); return 0; }
