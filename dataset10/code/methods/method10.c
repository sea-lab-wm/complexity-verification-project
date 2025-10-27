#include <stdio.h>

void method10() {
   if ( 1 && (! 0) ) {
      printf("true");
   } else {
      printf("false");
   }
}

int main() { method10(); return 0; }
