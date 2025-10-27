#include <stdio.h>

void method3() {
    // added to allow compilation
    int x,y,z,p;
    if( x++ ==y && x-- == z ) p=1;
    else p=2;
}

int main() { method3(); return 0; }
