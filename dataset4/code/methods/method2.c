#include <stdio.h>

void method2() {
    // added to allow compilation
    int x,y;
    x=y++;
}

int main() { method2(); return 0; }
