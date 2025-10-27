// Questions B - SE programs

// added to allow compilation
int x,y,z,i,p,m,n,a,b,j;

// Problem 1
void prob1() {
    y=a;
    x=y++ +b;
    while ((++y<=4)&&(!(x%2==0))) // changed = = to ==
    {
        ++x;
        if(!(y++%2==0)) x++; // changed = = to ==
    }
}

// Problem 2
void prob2() {
    for (i=a; i<5; i++)  
    {
        y=++i + a++;  
        i++;
    }
}

// Problem 3
void prob3() {
    for (i=a; i<n; i++)  
    { 
        m=i-1;  
        for (j=m; j<n; j++){ 
            if (!(j++==i) && (++b>1)) ++b;   // changed = = to ==
            b++; 
        }  
    }
}

// Problem 4
void prob4() {
    while (i <=7 ) 
    {
        a = ++i+b; 
        if (!(++i%3==0)) ++a; // changed = = to ==
        if(a%2==0) ++a; // changed = = to ==
    }
}

// added to allow compilation
int main() {
    return 0;
}