#include <stdio.h>

#define M1(V1, V2) V1 * V2

void method39(){
   int V3 = M1(1 + 2, 3 + 4);
   printf("%d\n", V3);
}

int main() { method39(); return 0; }
