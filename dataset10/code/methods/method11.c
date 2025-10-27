#include <stdio.h>

void method11() {
   if (0 && 1 || 2) {
      printf("true");
   } else {
      printf("false");
   }
}

int main() { method11(); return 0; }
