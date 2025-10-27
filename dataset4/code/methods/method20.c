#include <stdio.h>

void method20() {
    // added to allow compilation
    int a,b,i;
    while( i<=7 )
    {
        a=i+b+1;
        i=i+1;
        if( !((i+1)%3==0) ) a=a+1;
        i=i+1;
        if( a%2==0 ) a=a+1;
    }    
}

int main() { method20(); return 0; }
