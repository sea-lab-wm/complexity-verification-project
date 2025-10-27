#include <stdio.h>

void method35() {
   int V1 = 1, V2 = 2;
   if (V1 > V2)
   V2 = 1;
   V1 = 2;
   printf("%d %d\n",V1, V2);
}

int main() { method35(); return 0; }
