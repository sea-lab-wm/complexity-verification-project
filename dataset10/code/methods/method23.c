#include <stdio.h>

void method23() { 
   int V1 = 2;
   int V2 = --V1 + 3;
   printf("%d %d\n", V1, V2);
}

int main() { method23(); return 0; }
