#include <stdio.h>

void method24() { 
   int V1 = 6, V2;
   V2 = 9 - V1;
   --V1;
   printf("%d %d\n", V1, V2);
}

int main() { method24(); return 0; }
