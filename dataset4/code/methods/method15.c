#include <stdio.h>

void method15() {
    // added to allow compilation
    int x,y,a,b,j,m,n,i;
    for (i=a; i<n; i++)  
    { 
        m=i-1;  
        for (j=m; j<n; j++){ 
            if (!(j++==i) && (++b>1)) ++b;   // changed = = to ==
            b++; 
        }  
    }
}

int main() { method15(); return 0; }
