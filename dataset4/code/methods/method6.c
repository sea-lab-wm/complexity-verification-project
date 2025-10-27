#include <stdio.h>

void method6() {
    // added to allow compilation
    int x,i;
    if (i=0) i++; 
    else i--; 
    x=i--;
}

int main() { method6(); return 0; }
