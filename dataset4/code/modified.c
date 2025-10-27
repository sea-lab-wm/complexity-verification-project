// ID 1    prob1-t1se
void method1() {
    // added to allow compilation
    int x,y;
    x=++y;
}
// ID 2   prob2-t1se   
void method2() {
    // added to allow compilation
    int x,y;
    x=y++;
}
// ID 3 prob3-t1se
void method3() {
    // added to allow compilation
    int x,y,z,p;
    if( x++ ==y && x-- == z ) p=1;
    else p=2;
}
// ID 4 prob4-t1se
void method4() {
    // added to allow compilation
    int x,i;
    if (++i!=0) x=i++;
    else x=--i;
}
// ID 5 prob5-t1se
void method5() {
    // added to allow compilation
    int x,i;
    if(i++>0) {x=i++;}
    else {x=--i;}
    i=++x;
}
// ID 6 prob6-t1se
void method6() {
    // added to allow compilation
    int x,i;
    if (i=0) i++; 
    else i--; 
    x=i--;
}
// ID 7 prob1-t1sef
void method7() {
    // added to allow compilation
    int x,y;
    x=y+1;
    y=y+1;
}
// ID 8 prob2-t1sef
void method8() {
    // added to allow compilation
    int x,y;
    x=y;
    y=y+1;
}
// ID 9 prob3-t1sef
void method9() {
    // added to allow compilation
    int x,y,z,p;
    if( x==y&&(x+1)==z ) p=1;
    else if(x!=y) {x=x+1; p=2;}
}
// ID 10 prob4-t1sef
void method10() {
    // added to allow compilation
    int x,i;
    if(i != -1){ x=i+1; i=i+2; }
    else x=i;
}
// ID 11 prob5-t1sef
void method11() {
    // added to allow compilation
    int x,i;
    if(i>0) x=i+1;
    else x=i;
    i=x+1;
    x=x+1;
}
// ID 12 prob6-t1sef
void method12() {
    // added to allow compilation
    int x,i;
    x = -1;
    i = -2;
}
// ID 13 prob1-t2se
void method13() {
    // added to allow compilation
    int x,y,a,b;
    y=a;
    x=y++ +b;
    while ((++y<=4)&&(!(x%2==0))) // changed = = to ==
    {
        ++x;
        if(!(y++%2==0)) x++; // changed = = to ==
    }
}
// ID 14 prob2-t2se
void method14() {
    // added to allow compilation
    int x,y,a,b,i;
    for (i=a; i<5; i++)  
    {
        y=++i + a++;  
        i++;
    }
}
// ID 15 prob3-t2se
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
// ID 16 prob4-t2se
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
// ID 17 prob1-t2sef
void method17() {
    // added to allow compilation
    int x,y,a,b;
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
// ID 18 prob2-t2sef
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
// ID 19 prob3-t2sef
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
// ID 20 prob4-t2sef
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