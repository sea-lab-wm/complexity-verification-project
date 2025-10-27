#include <stdio.h>

void method8() {
    // added to allow compilation
    int x,y,z;
    x=y;
    y=y+1;
}

int main() { method8(); return 0; }
