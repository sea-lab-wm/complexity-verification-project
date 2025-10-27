#include <stdio.h>

void method12() {
   if ((2 && 0) || 5) {
      printf("true");
   } else {
      printf("false");
   }
}

int main() { method12(); return 0; }
