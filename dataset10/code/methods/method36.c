#include <stdio.h>

void method36() {
   int V1 = 5, V2 = 2;
   if (V1 < V2)
      V1 = 2;
   V2 = 5;
   printf("%d %d\n",V1, V2);
}

int main() { method36(); return 0; }
