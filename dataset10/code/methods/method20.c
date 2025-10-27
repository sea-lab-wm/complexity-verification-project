#include <stdio.h>

void method20() { 
   int V1 = 5, V2;
   ++V1;
   V2 = 5 - V1;
   printf("%d %d\n", V1, V2);
}

int main() { method20(); return 0; }
