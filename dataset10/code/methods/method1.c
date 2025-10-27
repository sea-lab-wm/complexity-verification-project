#include <stdio.h>

void method1(){
   int V1 = 10, V2 = 3;
   if (! (V1 % V2)){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}

int main() { method1(); return 0; }