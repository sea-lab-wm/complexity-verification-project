#include <stdio.h>

void method4()
{
   int V1 = 7;
   if ((V1 = 8) != 0)
      printf("true");
   else
      printf("false");
}

int main() { method4(); return 0; }
