#include <stdio.h>

// ID 1
void method1(){
   int V1 = 10, V2 = 3;
   if (! (V1 % V2)){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}
// ID 2
void method2(){
   int V1 = 1, V2 = 2;
   if ( (V2 - V1) == 0){
      printf("%s\n", "true");
   }
   else{
      printf("%s\n", "false");
   }
}
// ID 3
void method3() { 
   int V1 = 7; 
   if (V1 = 8) 
      printf("true\n"); 
   else
      printf("false\n"); 
}
// ID 4
void method4()
{
   int V1 = 7;
   if ((V1 = 8) != 0)
      printf("true");
   else
      printf("false");
}
// ID 5
void method5() {
   if ("V1")
      printf("true");
   else
      printf("false");
}
// ID 6
void method6() {
   if ("V1" != '\0')
      printf("true\n");
   else
      printf("false\n");
}
// ID 7
void method7() { 
   int V1;
   V1 = 2 - 4 / 2;
   printf("%d\n", V1);
}
// ID 8
void method8() { 
   int V1;
   V1 = 3 + (9 / 3);
   printf("%d\n", V1);
}
// ID 9
void method9() {
   if (! 0 && 2) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 10
void method10() {
   if ( 1 && (! 0) ) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 11
void method11() {
   if (0 && 1 || 2) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 12
void method12() {
   if ((2 && 0) || 5) {
      printf("true");
   } else {
      printf("false");
   }
}
// ID 13
void method13() { 
   int V1 = 2;
   int V2 = 3 + V1++;
   printf("%d %d\n", V1, V2);
}
// ID 14
void method14() { 
   int V1 =  2, V2;
   V2 = V1 + 3;
   V1++;
   printf("%d %d\n", V1, V2);
}
// ID 15
void method15() { 
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
void method16() { 
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
void method17() { 
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
void method18() { 
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
void method19() { 
   int V1 = 2;
   int V2 = ++V1 - 2;
   printf("%d %d\n", V1, V2);
}
// ID 20
void method20() { 
   int V1 = 5, V2;
   ++V1;
   V2 = 5 - V1;
   printf("%d %d\n", V1, V2);
}
// ID 21
void method21() { 
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
void method22() { 
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
void method23() { 
   int V1 = 2;
   int V2 = --V1 + 3;
   printf("%d %d\n", V1, V2);
}
// ID 24
void method24() { 
   int V1 = 6, V2;
   V2 = 9 - V1;
   --V1;
   printf("%d %d\n", V1, V2);
}
// ID 25
void method25() {
   int V1 = 3;
   int V2 = V1 + 2;
   printf("%d\n", V2);
}
// ID 26
void method26() {
   int V1 =  2 + 3;
   printf("%d\n", V1);
}
// ID 27
void method27() {
   int V1 = 2;
   int V2 = 2 * V1;
   printf("%d\n", V2);
}
// ID 28
void method28() {
   int V1 = 3 * 2;
   printf("%d\n", V1);
}
// ID 29
void method29() {
   int V1 = 2;
   printf("%f\n", 2.5 * V1);
}
// ID 30
void method30() {
   printf("%f\n", 2 * 4.5);
}
// ID 31
void method31() {
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
void method32() {
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
void method33() {
   int V1 = 5, V2 = 5;
   while (V2 > 0)
      V2--;
      V1++;
   printf("%d\n",V1);
}
// ID 34
void method34() {
   int V1 = 5, V2 = 5;
   while (V2 > 0)
      V2--;
   V1++;
   printf("%d\n",V1);
}
// ID 35
void method35() {
   int V1 = 1, V2 = 2;
   if (V1 > V2)
   V2 = 1;
   V1 = 2;
   printf("%d %d\n",V1, V2);
}
// ID 36
void method36() {
   int V1 = 5, V2 = 2;
   if (V1 < V2)
      V1 = 2;
   V2 = 5;
   printf("%d %d\n",V1, V2);
}
// ID 37
#define M1 64 - 1

void method37(){
   int V1;
   V1 = M1 * 2;
   printf("%d\n", V1);
}
// ID 38
void method38(){
   int V1;
   V1 = 128 - 1 * 2;
   printf("%d\n", V1);
}
// ID 39
#define M1(V1, V2) V1 * V2

void method39(){
   int V3 = M1(1 + 2, 3 + 4);
   printf("%d\n", V3);
}
// ID 40
void method40(){
   int V1 = 2 + 1 * 4 + 3;
   printf("%d\n", V1);
}
// ID 41
#define M1(V1, V2) (V1) * (V2)

void method41(){
   int V3 = M1(1 + 2, 3 + 4);
   printf("%d\n", V3);
}
// ID 42
void method42(){
   int V1 = (2 + 1) * (3 + 4);
   printf("%d\n", V1);
}
// ID 43
void method43() {
   char *V1 = "abcdef" + 3;
   printf("%s\n", V1);
}
// ID 44
void method44() {
   char *V1 = &("abcdef"[2]);
   printf("%s\n", V1);
}
// ID 45
void method45() {
   int V1[] = {4, 2, 7, 5};
   int *V2 = V1 + 1;
   printf("%d\n", *V2);
}
// ID 46
void method46() {
   int V1[] = {3, 1, 4, 6};
   int *V2 = &V1[1];
   printf("%d\n", *V2);
}
// ID 47
void method47() {
   int V1[] = {4, 7, 2, 3};
   int *V2 = V1 + 1;
   V2 = V2 + 2;
   printf("%d\n", *V2);
}
// ID 48
void method48() {
   int V1[] = {3, 2, 9, 4};
   int *V2 = &V1[1];
   V2 = &V2[2];
   printf("%d\n", *V2);
}
// ID 49
void method49() {
   int V1 = 4;

   int V2 = V1 == 3 ? 2 : 1;

   printf("%d\n", V2);
}
// ID 50
void method50() {
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
void method51() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4 = (V1 == 2 ? (V3 == 2 ? 1 : 2) : (V2 == 2 ? 3 : 4));

   printf("%d\n", V4);
}
// ID 52
void method52() {
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
void method53() {
   int V1 = 2;
   int V2 = 3;
   int V3 = 1;

   int V4 = V1 == 3 ? V2 : V3;

   printf("%d\n", V4);
}
// ID 54
void method54() {
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
void method55() {
   int V1 = 8;

   if ((V1 - 3) * (7 - V1) <= 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 56
void method56() {
   int V1 = 8;

   if (3 <= V1 || V1 >= 7) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 57
void method57() {
   int V1 = 2;

   if ((V1 - 2) * (6 - V1) > 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 58
void method58() {
   int V1 = 2;

   if (V1 < 2 || 6 < V1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 59
void method59() {
   int V1 = 5;

   if (V1 + 5 != 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 60
void method60() {
   int V1 = 5;

   if (V1 != -5) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 61
void method61() {
   int V1, V2;

   V1 = (V2 = 1, 2);

   printf("%d %d\n", V1, V2);
}
// ID 62
void method62() {
   int V1, V2;

   V1 = 2;
   V2 = 1;

   printf("%d %d\n", V1, V2);
}
// ID 63
void method63() {
   int V1 = 3;
   int V2 = (V1 *= 2, V1 += 1);

   printf("%d %d\n", V1, V2);
}
// ID 64
void method64() {
   int V1 = 3;
  
   V1 *= 2;

   int V2 = (V1 += 1);

   printf("%d %d\n", V1, V2);
}
// ID 65
void method65() {
   int V1 = 3;
   int V2 = (V1 += 1, V1 *= 2);

   printf("%d %d\n", V1, V2);
}
// ID 66
void method66() {
   int V1 = 3;
  
   V1 += 1;

   int V2 = (V1 *= 2);

   printf("%d %d\n", V1, V2);
}
// ID 67
void method67() {
   char *V1 = "abcdef"
   #define M1
   "abcdef";

   printf("%s\n", V1);
}
// ID 68
void method68() {
   char *V1 = "abcdef"
   "abcdef";

   #define M1

   printf("%s\n", V1);
}
// ID 69
void method69() {
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
void method70() {
   #define M1 1
   #define M2 2

   int V1;
   V1 = 4;

   int V2 = 1 + V1;

   printf("%d %d\n", V1, V2);
}
// ID 71
void method71() {
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
void method72() {
   int V1 = 1, V2 = 2;
   #define M1 2
   #define M2 1

   printf("%d %d\n", M1, M2);
}
// ID 73
void method73() {
   int V1 = 2;

   if (V1 = 1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 74
void method74() {
   int V1 = 7;
   V1 = 1;
   if (1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 75
void method75() {
   int V1 = 0;

   if (V1 = 0) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 76
void method76() {
   int V1 = 0;

   V1 = 0;

   if (V1) {
      printf("true\n");
   } else {
      printf("false\n");
   }
}
// ID 77
void method77() {
   int V1 = 0;
   int V2 = 9;

   while (!(V1 = 3)) {
      V2--;
      V1++;
   }

   printf("%d %d\n", V1, V2);
}
// ID 78
void method78() {
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
void method79() {
   int V1 = 1;
   int V2 = 5;

   if (++V1 || ++V2) {
      V1 = V1 * 2;
      V2 = V2 * 2;
   }

   printf("%d %d\n", V1, V2);
}
// ID 80
void method80() {
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
void method81() {
   int V1 = 1;
   int V2 = 5;

   V1 == V2 && ++V1 || ++V2;

   printf("%d %d\n", V1, V2);
}
// ID 82
void method82() {
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
void method83() {
   int V1 = 3;
   int V2 = 5;
   int V3 = 0;

   while (V1 != V2 && ++V1) {
      V3++;
   }

   printf("%d %d %d\n", V1, V2, V3);
}
// ID 84
void method84() {
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
void method85() {
   int V1[5];
   V1[4] = 3;

   while (V1[4]) {
      V1[3 - V1[4]] = V1[4];
      V1[4] = V1[4] - 1;
   }

   printf("%d %d\n", V1[1], V1[4]);
}
// ID 86
void method86() {
   int V1[6];
   int V2 = 5;

   while (V2) {
      V1[5 - V2] = V2;
      V2 = V2 - 1;
   }

   printf("%d %d\n", V1[1], V2);
}
// ID 87
void method87() {
   int V3 = 0;

   for (int V1 = 0; V1 < 2; V1++) {
      for (int V2 = 0; V1 < 2; V1++) {
        V3 = 4 * V1 + V2;
      }
   }

   printf("%d\n", V3);
}
// ID 88
void method88() {
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
void method89() {
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
void method90() {
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
void method91() {
   char V1 = 2["qwert"];

   printf("%c\n", V1);
}
// ID 92
void method92() {
   char V1 = "zxcvb"[4];

   printf("%c\n", V1);
}
// ID 93
void method93() {
   char V1 = 3;
   char V2 = V1["zxcvb"];

   printf("%c\n", V2);
}
// ID 94
void method94() {
   char V1 = 2;
   char V2 = "asdfg"[V1];

   printf("%c\n", V2);
}
// ID 95
void method95() {
   char V1 = 4;
   char* V2 = "qazwsx";
   char V3 = V1[V2];

   printf("%c\n", V3);
}
// ID 96
void method96() {
   char V1 = 4;
   char* V2 = "abcdef";
   char V3 = V2[V1];

   printf("%c\n", V3);
}
// ID 97
void method97() {
   int V1 = 1;

   V1 = 3;
   V1 = 2;

   printf("%d\n", V1);
}
// ID 98
void method98() {
  int V1 = 1;

  V1 = 2;

  printf("%d\n", V1);
}
// ID 99
void method99() {
   int V1 = 1;

   if (0) {
      V1 = 3;
   }

   printf("%d\n", V1);
}
// ID 100
void method100() {
   int V1 = 1;

   printf("%d\n", V1);
}
// ID 101
void method101() {
   int V1 = 0;

   V1 = V1;

   printf("%d\n", V1);
}
// ID 102
void method102() {
   int V1 = 0;

   printf("%d\n", V1);
}
// ID 103
void method103() {
   char V1 = 104;
   printf("%c\n", V1);
}
// ID 104
void method104() {
   char V1 = 'g';

   printf("%c\n", V1);
}
// ID 105
void method105() {
   int V1 = 013;

   printf("%d\n", V1);
}
// ID 106
void method106() {
   char V1 = 23;
 
   printf("%d\n", V1);
}
// ID 107
void method107() {
   int V1 = 208 & 13;

   printf("%d\n", V1);
}
// ID 108
void method108() {
   char V1 = 0xD0 & 0x0D;

   printf("%d\n", V1);
}
// ID 109
void method109() {
   int V1 = 2;

   if (0) V1++; V1++;

   printf("%d\n", V1);
}
// ID 110
void method110() {
   int V1 = 2;

   if (0) { V1++; } V1++;

   printf("%d\n", V1);
}
// ID 111
void method111() {
   int V1 = 4;

   int V2 = 0;
   while (V2 < 3) V2++; V1++;

   printf("%d %d\n", V1, V2);
}
// ID 112
void method112() {
   int V1 = 7;

   int V2 = 1;
   while (V2 < 3) { V2++; } V1++;
 
   printf("%d %d\n", V1, V2);
}
// ID 113
void method113() {
   int V1 = 3;

   for (int V2 = 0; V2 < 3; V2++) V1++; V1++;

   printf("%d\n", V1);
}
// ID 114
void method114() {
   int V1 = 4;

   for (int V2 = 0; V2 < 3; V2++) { V1++; } V1++;

   printf("%d\n", V1);
}
// ID 115
void method115() {
   float V1 = 1.99;

   int V2 = V1;
 
   printf("%d\n", V2);
}
// ID 116
#include <math.h>
void method116() {
   float V1 = 2.87;

   int V2 = trunc(V1);

   printf("%d\n", V2);
}
// ID 117
void method117() {
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
void method118() {
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
void method119() {
   int V1 = 261;

   char V2 = V1;

   printf("%d\n", V2);
}
// ID 120
void method120() {
   int V1 = 288;

   char V2 = V1 % 256;

   printf("%d\n", V2);
}
// ID 121
void method121() {
   int V1 = 0;
   int V2 = 2;

   if (V1) {}
      V2 = 4;

   printf("%d\n", V2);
}
// ID 122
void method122() {
   int V1 = 0;
   int V2 = 2;

   if (V1) {}
   V2 = 4;

   printf("%d\n", V2);
}
// ID 123
void method123() {
   int V1 = 0;
   int V2 = 1;

   if (V1) {
      V2 = 2;
   }
      V2 = V2 * 3;

   printf("%d\n", V2);
}
// ID 124
void method124() {
   int V1 = 0;
   int V2 = 5;

   if (V1) {
      V2 = 2;
   }
   V2 = V2 * 2;

   printf("%d\n", V2);
}
// ID 125
void method125() {
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
void method126() {
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
int main() {
    return 0;
}