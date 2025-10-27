#include <stdio.h>

void method14() { 
   int V1 =  2, V2;
   V2 = V1 + 3;
   V1++;
   printf("%d %d\n", V1, V2);
}

int main() { method14(); return 0; }
