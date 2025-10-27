#include <stdio.h>

void method2(){
   int V1 = 1, V2 = 2;
   if ( (V2 - V1) == 0){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}

int main() { method2(); return 0; }
