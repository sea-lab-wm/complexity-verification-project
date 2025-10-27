#include <stdio.h>

void method13() { 
   int V1 = 2;
   int V2 = 3 + V1++;
   printf("%d %d\n", V1, V2);
}

int main() { method13(); return 0; }
