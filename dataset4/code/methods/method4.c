#include <stdio.h>

void method4() {
    // added to allow compilation
    int x,i;
    if (++i!=0) x=i++;
    else x=--i;
}

int main() { method4(); return 0; }
