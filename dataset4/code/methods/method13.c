#include <stdio.h>

void method13() {
    // added to allow compilation
    int x,y,a,b;
    y=a;
    x=y++ +b;
    while ((++y<=4)&&(!(x%2==0))) // changed = = to ==
    {
        ++x;
        if(!(y++%2==0)) x++; // changed = = to ==
    }
}

int main() { method13(); return 0; }
