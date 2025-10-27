#include <stdio.h>

void method1() {
    // added to allow compilation
    int x,y;
    x=++y;
}

int main() { method1(); return 0; }
