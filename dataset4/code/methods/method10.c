#include <stdio.h>

void method10() {
    // added to allow compilation
    int x,i;
    if(i != -1){ x=i+1; i=i+2; }
    else x=i;
}

int main() { method10(); return 0; }
