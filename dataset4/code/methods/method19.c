#include <stdio.h>

void method19() {
    // added to allow compilation
    int x,y,a,b,j,m,n,i;
    i=a;
    for(;i<n;)
    {
        m=i-1;
        j=m;
        for(;j<n;)
        {
            if( !(j==i) && (b>0) ) b=b+2;
            else if(!(j==i)) b=b+1;
            b=b+1;
            j=j+2;
        }
        i=i+1;
    }
}

int main() { method19(); return 0; }
