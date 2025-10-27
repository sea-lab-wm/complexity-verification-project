#include <stdio.h>

void method61() {
   int V1, V2;

   V1 = (V2 = 1, 2);

   printf("%d %d\n", V1, V2);
}

int main() { method61(); return 0; }
