// Questions A - SE programs


// added to allow compilation
int x,y,z,i,p;

// Problem 1
void prob1() {
    x=++y;
}

// Problem 2
void prob2() {
    x=y++;
}

// Problem 3
void prob3() {
    if( x++ ==y && x-- == z ) p=1;
    else p=2;
}

// Problem 4
void prob4() {
    if (++i!=0) x=i++;
    else x=--i;
}

// Problem 5
void prob5() {
    if(i++>0) {x=i++;}
    else {x=--i;}
    i=++x;
}

// Problem 6
void prob6() {
    if (i=0) i++; 
    else i--; 
    x=i--;
}

