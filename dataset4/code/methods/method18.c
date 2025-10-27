#include <stdio.h>

void method18() {
    // added to allow compilation
    int x,y,a,b,i;
    i=a;
    for(;i<5;)
    {
        y=i+a+1;
        i=i+3;
        a=a+1;
    }
}

int main() { method18(); return 0; }
