#include <stdio.h>

void method95() {
   char V1 = 4;
   char* V2 = "qazwsx";
   char V3 = V1[V2];

   printf("%c\n", V3);
}

int main() { method95(); return 0; }
