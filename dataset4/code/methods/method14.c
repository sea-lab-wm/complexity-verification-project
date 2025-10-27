#include <stdio.h>

void method14() {
    // added to allow compilation
    int x,y,a,b,i;
    for (i=a; i<5; i++)  
    {
        y=++i + a++;  
        i++;
    }
}

int main() { method14(); return 0; }
