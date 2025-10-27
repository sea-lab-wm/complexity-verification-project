#include <stdio.h>

void method47() {
   int V1[] = {4, 7, 2, 3};
   int *V2 = V1 + 1;
   V2 = V2 + 2;
   printf("%d\n", *V2);
}

int main() { method47(); return 0; }
