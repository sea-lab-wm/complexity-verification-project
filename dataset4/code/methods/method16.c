#include <stdio.h>

void method16() {
    // added to allow compilation
    int a,b,i;
    while (i <=7 ) 
    {
        a = ++i+b; 
        if (!(++i%3==0)) ++a; // changed = = to ==
        if(a%2==0) ++a; // changed = = to ==
    }
}

int main() { method16(); return 0; }
