// ID 1
void main(){
   int V1 = 10, V2 = 3;
   if (! (V1 % V2)){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}
// ID 2
void main(){
   int V1 = 1, V2 = 2;
   if ( (V2 - V1) == 0){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}
// ID 3
void main() { 
   int V1 = 7; 
   if (V1 = 8) 
      printf("true\n"); 
   else
      printf("false\n"); 
}
// ID 4
void main()
{
   int V1 = 7;
   if ((V1 = 8) != 0)
      printf("true");
   else
      printf("false");
}
// ID 5
void main() {
   if ("V1")
      printf("true");
   else
      printf("false");
}
// ID 6
void main() {
   if ("V1" != '\0')
      printf("true\n");
   else
      printf("false\n");
}
// ID 7
void main() { 
   int V1;
   V1 = 2 - 4 / 2;
   printf("%d\n", V1);
}
// ID 8
void main() { 
   int V1;
   V1 = 3 + (9 / 3);
   printf("%d\n", V1);
}
// ID 9
void main() {
   if (! 0 && 2) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 10
void main() {
   if ( 1 && (! 0) ) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 11
void main() {
   if (0 && 1 || 2) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 12
void main() {
   if ((2 && 0) || 5) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 13
void main() { 
   int V1 = 2;
   int V2 = 3 + V1++;
   printf("%d %d\n", V1, V2);
}
// ID 14
void main() { 
   int V1 =  2, V2;
   V2 = V1 + 3;
   V1++;
   printf("%d %d\n", V1, V2);
}
// ID 15
void main() { 
   int V1 = 0;
   if (V1++ == 0) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}
// ID 16
void main() { 
   int V1 = 0;
   if (V1 == 0) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   V1++;
   printf("%d", V1);
}
// ID 17
void main() { 
   int V1 = 2;
   if (V1-- == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}
// ID 18
void main() { 
   int V1 = 2;
   if (V1 == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   V1--;
   printf("%d", V1);
}
// ID 19
void main() { 
   int V1 = 2;
   int V2 = ++V1 - 2;
   printf("%d %d\n", V1, V2);
}
// ID 20
void main() { 
   int V1 = 5, V2;
   ++V1;
   V2 = 5 - V1;
   printf("%d %d\n", V1, V2);
}
// ID 21
void main() { 
   int V1 = 2;
   if (--V1 == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}
// ID 22
void main() { 
   int V1 = 2;
   V1--;
   if (V1 == 1) {
      printf("true ");
   }
   else {
      printf("false ");
   }
   printf("%d", V1);
}
// ID 23
void main() { 
   int V1 = 2;
   int V2 = --V1 + 3;
   printf("%d %d\n", V1, V2);
}
// ID 24
void main() { 
   int V1 = 6, V2;
   V2 = 9 - V1;
   --V1;
   printf("%d %d\n", V1, V2);
}
// ID 25
void main() {
   int V1 = 3;
   int V2 = V1 + 2;
   printf("%d\n", V2);
}
// ID 26
void main() {
   int V1 =  2 + 3;
   printf("%d\n", V1);
}
// ID 27
void main() {
   int V1 = 2;
   int V2 = 2 * V1;
   printf("%d\n", V2);
}
// ID 28
void main() {
   int V1 = 3 * 2;
   printf("%d\n", V1);
}
// ID 29
void main() {
   int V1 = 2;
   printf("%f\n", 2.5 * V1);
}
// ID 30
void main() {
   printf("%f\n", 2 * 4.5);
}
// ID 31
void main() {
   int V1 = 2;
   int V2 = 0;
   int V3 = 3;

   if (V1)
      if (V2)
         V3 = V3 + 2;
   else
      V3 = V3 + 4;

   printf("%d\n", V3);
}
// ID 32
void main() {
   int V1 = 2;
   int V2 = 0;
   int V3 = 5;

   if (V1)
      if (V2)
         V3 = V3 + 2;
      else
         V3 = V3 + 4;

   printf("%d\n", V3);
}
// ID 33
void main() {
   int V1 = 5, V2 = 5;
   while (V2 > 0)
      V2--;
      V1++;
   printf("%d\n",V1);
}
// ID 34
void main() {
   int V1 = 5, V2 = 5;
   while (V2 > 0)
      V2--;
   V1++;
   printf("%d\n",V1);
}
// ID 35
void main() {
   int V1 = 1, V2 = 2;
   if (V1 > V2)
   V2 = 1;
   V1 = 2;
   printf("%d %d\n",V1, V2);
}
// ID 36
void main() {
   int V1 = 5, V2 = 2;
   if (V1 < V2)
      V1 = 2;
   V2 = 5;
   printf("%d %d\n",V1, V2);
}
// ID 37
#define M1 64 - 1

void main(){
   int V1;
   V1 = M1 * 2;
   printf("%d\n", V1);
}
// ID 38
void main(){
   int V1;
   V1 = 128 - 1 * 2;
   printf("%d\n", V1);
}
// ID 39
#define M1(V1, V2) V1 * V2

void main(){
   int V3 = M1(1 + 2, 3 + 4);
   printf("%d\n", V3);
}
// ID 40
void main(){
   int V1 = 2 + 1 * 4 + 3;
   printf("%d\n", V1);
}
// ID 41
#define M1(V1, V2) (V1) * (V2)

void main(){
   int V3 = M1(1 + 2, 3 + 4);
   printf("%d\n", V3);
}
// ID 42
void main(){
   int V1 = (2 + 1) * (3 + 4);
   printf("%d\n", V1);
}
// ID 43
void main() {
   char *V1 = "abcdef" + 3;
   printf("%s\n", V1);
}
// ID 44
void main() {
   char *V1 = &("abcdef"[2]);
   printf("%s\n", V1);
}
// ID 45
void main() {
   int V1[] = {4, 2, 7, 5};
   int *V2 = V1 + 1;
   printf("%d\n", *V2);
}
// ID 46
void main() {
   int V1[] = {3, 1, 4, 6};
   int *V2 = &V1[1];
   printf("%d\n", *V2);
}
// ID 47
void main() {
   int V1[] = {4, 7, 2, 3};
   int *V2 = V1 + 1;
   V2 = V2 + 2;
   printf("%d\n", *V2);
}
// ID 48
void main() {
   int V1[] = {3, 2, 9, 4};
   int *V2 = &V1[1];
   V2 = &V2[2];
   printf("%d\n", *V2);
}
// ID 49
void main() {
   int V1 = 4;

   int V2 = V1 == 3 ? 2 : 1;

   printf("%d\n", V2);
}
// ID 50
void main() {
   int V1 = 4;
   int V2 = 3;
   int V3;
  
   if (V1 == 3) {
      V3 = 2;
   } else {
      V3 = 1;
   }

   printf("%d\n", V3);
}
// ID 51
void main() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4 = (V1 == 2 ? (V3 == 2 ? 1 : 2) : (V2 == 2 ? 3 : 4));

   printf("%d\n", V4);
}
// ID 52
void main() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4;
   if (V1 == 2) {
      if (V3 == 2) {
         V4 = 1;
      } else {
         V4 = 2;
      }
   } else {
      if (V2 == 2) {
         V4 = 3;
      } else {
         V4 = 4;
      }
   }

   printf("%d\n", V4);
}
// ID 53
void main() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4 = V1 == 3 ? V2 : V3;

   printf("%d\n", V4);
}
// ID 54
void main() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4;
   if (V1 == 3){
      V4 = V2;
   }
   else{
      V4 = V3;
   }

   printf("%d\n", V4);
}
// ID 55
void main() {
   int V1 = 8;

   if ((V1 - 3) * (7 - V1) <= 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 56
void main() {
   int V1 = 8;

   if (3 <= V1 || V1 >= 7) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 57
void main() {
   int V1 = 2;

   if ((V1 - 2) * (6 - V1) > 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 58
void main() {
   int V1 = 2;

   if (V1 < 2 || 6 < V1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 59
void main() {
   int V1 = 5;

   if (V1 + 5 != 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 60
void main() {
   int V1 = 5;

   if (V1 != -5) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 61
void main() {
   int V1, V2;

   V1 = (V2 = 1, 2);

   printf("%d %d\n", V1, V2);
}
// ID 62
void main() {
   int V1, V2;

   V1 = 2;
   V2 = 1;

   printf("%d %d\n", V1, V2);
}
// ID 63
void main() {
   int V1 = 3;
   int V2 = (V1 *= 2, V1 += 1);

   printf("%d %d\n", V1, V2);
}
// ID 64
void main() {
   int V1 = 3;
  
   V1 *= 2;

   int V2 = (V1 += 1);

   printf("%d %d\n", V1, V2);
}
// ID 65
void main() {
   int V1 = 3;
   int V2 = (V1 += 1, V1 *= 2);

   printf("%d %d\n", V1, V2);
}
// ID 66
void main() {
   int V1 = 3;
  
   V1 += 1;

   int V2 = (V1 *= 2);

   printf("%d %d\n", V1, V2);
}
// ID 67
void main() {
   char *V1 = "abcdef"
   #define M1
   "abcdef";

   printf("%s\n", V1);
}
// ID 68
void main() {
   char *V1 = "abcdef"
   "abcdef";

   #define M1

   printf("%s\n", V1);
}
// ID 69
void main() {
   int V1;
   V1 = 4;

   int V2 = 1
   #define M1 1
   +
   #define M2 2
   V1;

   printf("%d %d\n", V1, V2);
}
// ID 70
void main() {
   #define M1 1
   #define M2 2

   int V1;
   V1 = 4;

   int V2 = 1 + V1;

   printf("%d %d\n", V1, V2);
}
// ID 71
void main() {
   int V1 = 1, V2 = 2;

   if (V1 < V2) {
      #define M1 1
      #define M2 2
   } else {
      #define M1 2
      #define M2 1
   }

   printf("%d %d\n", M1, M2);
}
// ID 72
void main() {
   int V1 = 1, V2 = 2;
   #define M1 2
   #define M2 1

   printf("%d %d\n", M1, M2);
}
// ID 73
void main() {
   int V1 = 2;

   if (V1 = 1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 74
void main() {
   int V1 = 7;
   V1 = 1;
   if (1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 75
void main() {
   int V1 = 0;

   if (V1 = 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 76
void main() {
   int V1 = 0;

   V1 = 0;

   if (V1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 77
void main() {
   int V1 = 0;
   int V2 = 9;

   while (!(V1 = 3)) {
      V2--;
      V1++;
   }

   printf("%d %d\n", V1, V2);
}
// ID 78
void main() {
   int V1 = 0;
   int V2 = 7;

   V1 = 4;

   while (!4) {
      V2--;
      V1++;
   }

   printf("%d %d\n", V1, V2);
}
// ID 79
void main() {
   int V1 = 1;
   int V2 = 5;

   if (++V1 || ++V2) {
      V1 = V1 * 2;
      V2 = V2 * 2;
   }

   printf("%d %d\n", V1, V2);
}
// ID 80
void main() {
   int V1 = 2;
   int V2 = 4;

   if (++V1) {
        V1 = V1 * 2;
        V2 = V2 * 2;
   } else if (++V2) {
        V1 = V1 * 2;
        V2 = V2 * 2;
   }

   printf("%d %d\n", V1, V2);
}
// ID 81
void main() {
   int V1 = 1;
   int V2 = 5;

   V1 == V2 && ++V1 || ++V2;

   printf("%d %d\n", V1, V2);
}
// ID 82
void main() {
   int V1 = 2;
   int V2 = 6;

   if (V1 == V2) {
      ++V1;
   } else {
      ++V2;
   }

   printf("%d %d\n", V1, V2);
}
// ID 83
void main() {
   int V1 = 3;
   int V2 = 5;
   int V3 = 0;

   while (V1 != V2 && ++V1) {
      V3++;
   }

   printf("%d %d %d\n", V1, V2, V3);
}
// ID 84
void main() {
   int V1 = 1;
   int V2 = 11;
   int V3 = 0;

   while (V1 != V2) {
      ++V1;
      if (!V1) break;

      V3++;
   }

   printf("%d %d %d\n", V1, V2, V3);
}
// ID 85
void main() {
   int V1[5];
   V1[4] = 3;

   while (V1[4]) {
      V1[3 - V1[4]] = V1[4];
      V1[4] = V1[4] - 1;
   }

   printf("%d %d\n", V1[1], V1[4]);
}
// ID 86
void main() {
   int V1[6];
   int V2 = 5;

   while (V2) {
      V1[5 - V2] = V2;
      V2 = V2 - 1;
   }

   printf("%d %d\n", V1[1], V2);
}
// ID 87
void main() {
   int V3 = 0;

   for (int V1 = 0; V1 < 2; V1++) {
      for (int V2 = 0; V1 < 2; V1++) {
        V3 = 4 * V1 + V2;
      }
   }

   printf("%d\n", V3);
}
// ID 88
void main() {
   int V3 = 0;

   for (int V1 = 0; V1 < 2; V1++) {
      for (int V2 = 0; V2 < 2; V2++) {
         V3 = 4 * V1 + V2;
         V1 = V2;
      }
   }

   printf("%d\n", V3);
}
// ID 89
void main() {
   int V1;
   for (int V2 = 0; V2 < 2; V2++) {
      V1 = (V2 < 1);
      if (V1) {
         V1 = V2 + 5;
      } else {
         V1 = V1 + 2;
      }
   }
   printf("%d\n", V1);
}
// ID 90
void main() {
   int V1;
   for (int V2 = 0; V2 < 2; V2++) {
      int V3 = (V2 < 1);
      if (V3) {
         V1 = V2 + 4;
      } else {
         V1 = V3 + 4;
      }
   }
   printf("%d\n", V1);
}
// ID 91
void main() {
   char V1 = 2["qwert"];

   printf("%c\n", V1);
}
// ID 92
void main() {
   char V1 = "zxcvb"[4];

   printf("%c\n", V1);
}
// ID 93
void main() {
   char V1 = 3;
   char V2 = V1["zxcvb"];

   printf("%c\n", V2);
}
// ID 94
void main() {
   char V1 = 2;
   char V2 = "asdfg"[V1];

   printf("%c\n", V2);
}
// ID 95
void main() {
   char V1 = 4;
   char* V2 = "qazwsx";
   char V3 = V1[V2];

   printf("%c\n", V3);
}
// ID 96
void main() {
   char V1 = 4;
   char* V2 = "abcdef";
   char V3 = V2[V1];

   printf("%c\n", V3);
}
// ID 97
void main() {
   int V1 = 1;

   V1 = 3;
   V1 = 2;

   printf("%d\n", V1);
}
// ID 98
void main() {
  int V1 = 1;

  V1 = 2;

  printf("%d\n", V1);
}
// ID 99
void main() {
   int V1 = 1;

   if (0) {
      V1 = 3;
   }

   printf("%d\n", V1);
}
// ID 100
void main() {
   int V1 = 1;

   printf("%d\n", V1);
}
// ID 101
void main() {
   int V1 = 0;

   V1 = V1;

   printf("%d\n", V1);
}
// ID 102
void main() {
   int V1 = 0;

   printf("%d\n", V1);
}
// ID 103
void main() {
   char V1 = 104;
   printf("%c\n", V1);
}
// ID 104
void main() {
   char V1 = 'g';

   printf("%c\n", V1);
}
// ID 105
void main() {
   int V1 = 013;

   printf("%d\n", V1);
}
// ID 106
void main() {
   char V1 = 23;
 
   printf("%d\n", V1);
}
// ID 107
void main() {
   int V1 = 208 & 13;

   printf("%d\n", V1);
}
// ID 108
void main() {
   char V1 = 0xD0 & 0x0D;

   printf("%d\n", V1);
}
// ID 109
void main() {
   int V1 = 2;

   if (0) V1++; V1++;

   printf("%d\n", V1);
}
// ID 110
void main() {
   int V1 = 2;

   if (0) { V1++; } V1++;

   printf("%d\n", V1);
}
// ID 111
void main() {
   int V1 = 4;

   int V2 = 0;
   while (V2 < 3) V2++; V1++;

   printf("%d %d\n", V1, V2);
}
// ID 112
void main() {
   int V1 = 7;

   int V2 = 1;
   while (V2 < 3) { V2++; } V1++;
 
   printf("%d %d\n", V1, V2);
}
// ID 113
void main() {
   int V1 = 3;

   for (int V2 = 0; V2 < 3; V2++) V1++; V1++;

   printf("%d\n", V1);
}
// ID 114
void main() {
   int V1 = 4;

   for (int V2 = 0; V2 < 3; V2++) { V1++; } V1++;

   printf("%d\n", V1);
}
// ID 115
void main() {
   float V1 = 1.99;

   int V2 = V1;
 
   printf("%d\n", V2);
}
// ID 116
#include <math.h>
void main() {
   float V1 = 2.87;

   int V2 = trunc(V1);

   printf("%d\n", V2);
}
// ID 117
void main() {
   int V1 = -1;

   unsigned int V2 = V1;

   int V3;
   if (V2 > 0) {
      V3 = 4;
   } else {
      V3 = 5;
   }

   printf("%d\n", V3);
}
// ID 118
#include  <limits.h>
void main() {
   int V1 = -1;

   unsigned int V2;
   if (V1 >= 0) {
      V2 = V1;
   } else {
      V2 = UINT_MAX + (V1 + 1);
   }

   int V3;
   if (V2 >= 0) {
      V3 = 4;
   } else {
      V3 = 5;
   }  

   printf("%d\n", V3);
}
// ID 119
void main() {
   int V1 = 261;

   char V2 = V1;

   printf("%d\n", V2);
}
// ID 120
void main() {
   int V1 = 288;

   char V2 = V1 % 256;

   printf("%d\n", V2);
}
// ID 121
void main() {
   int V1 = 0;
   int V2 = 2;

   if (V1) {}
      V2 = 4;

   printf("%d\n", V2);
}
// ID 122
void main() {
   int V1 = 0;
   int V2 = 2;

   if (V1) {}
   V2 = 4;

   printf("%d\n", V2);
}
// ID 123
void main() {
   int V1 = 0;
   int V2 = 1;

   if (V1) {
      V2 = 2;
   }
      V2 = V2 * 3;

   printf("%d\n", V2);
}
// ID 124
void main() {
   int V1 = 0;
   int V2 = 5;

   if (V1) {
      V2 = 2;
   }
   V2 = V2 * 2;

   printf("%d\n", V2);
}
// ID 125
void main() {
   int V1 = 2;
   int V2 = 0;
   int V3 = 3;

   if (V1) {
      if (V2) {
         V3 = V3 +2;
   } else {
      V3 = V3 + 4;
   }
   }

   printf("%d\n", V3);
}
// ID 126
void main() {
  int V1 = 2;
  int V2 = 0;
  int V3 = 5;

  if (V1) {
     if (V2) {
        V3 = V3 + 2;
     } else {
        V3 = V3 + 4;
     }
  }

  printf("%d\n", V3);
}