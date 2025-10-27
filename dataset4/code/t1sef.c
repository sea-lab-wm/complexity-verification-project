// Questions A - SEF programs

// added to allow compilation
int x,y,z,i,p;

// Problem 1
void prob1() {
    x=y+1;
    y=y+1;
}

// Problem 2
void prob2() {
    x=y;
    y=y+1;
}

// Problem 3
void prob3() {
    if( x==y&&(x+1)==z ) p=1;
    else if(x!=y) {x=x+1; p=2;}
}

// Problem 4
void prob4() {
    if(i != -1){ x=i+1; i=i+2; }
    else x=i;
}

// Problem 5
void prob5() {
    if(i>0) x=i+1;
    else x=i;
    i=x+1;
    x=x+1;
}

// Problem 6
void prob6() {
    x = -1;
    i = -2;
}

// added to allow compilation
int main() {
    return 0;
}