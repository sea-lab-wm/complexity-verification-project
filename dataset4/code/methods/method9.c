#include <stdio.h>

void method9() {
    // added to allow compilation
    int x,y,z,p;
    if( x==y&&(x+1)==z ) p=1;
    else if(x!=y) {x=x+1; p=2;}
}

int main() { method9(); return 0; }
