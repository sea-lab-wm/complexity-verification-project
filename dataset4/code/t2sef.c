// Questions B - SEF programs

// added to allow compilation
int x,y,z,i,p,m,n,a,b,j;

// Problem 1
void prob1() {
    x=a+b;
    y=a+1;
    while( (y<=3)&&(!(x%2==0)) )
    {
        y=y+1;
        x=x+1;
        if( !(y%2==0) ) x=x+1; // changed = = to ==
        y=y+1;
    }
    y=y+1; // missing ; was added.
}

// Problem 2
void prob2() {
    i=a;
    for(;i<5;)
    {
        y=i+a+1;
        i=i+3;
        a=a+1;
    }
}

// Problem 3
void prob3() {
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

// Problem 4
void prob4() {
    while( i<=7 )
    {
        a=i+b+1;
        i=i+1;
        if( !((i+1)%3==0) ) a=a+1;
        i=i+1;
        if( a%2==0 ) a=a+1;
    }    
}