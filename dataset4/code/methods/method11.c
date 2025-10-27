#include <stdio.h>

void method11() {
    // added to allow compilation
    int x,i;
    if(i>0) x=i+1;
    else x=i;
    i=x+1;
    x=x+1;
}

int main() { method11(); return 0; }
