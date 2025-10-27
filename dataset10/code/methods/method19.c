#include <stdio.h>

void method19() { 
   int V1 = 2;
   int V2 = ++V1 - 2;
   printf("%d %d\n", V1, V2);
}

int main() { method19(); return 0; }
