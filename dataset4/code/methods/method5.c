#include <stdio.h>

void method5() {
    // added to allow compilation
    int x,i;
    if(i++>0) {x=i++;}
    else {x=--i;}
    i=++x;
}

int main() { method5(); return 0; }
