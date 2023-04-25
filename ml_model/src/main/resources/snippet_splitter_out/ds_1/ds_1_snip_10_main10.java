package FeatureExtraction.snippet_splitter_out;
public class ds_1_snip_10_main10 {
// 10. Find largest number of three numbers                                      /*Tasks for fMRI-Setting*/
// SNIPPET_STARTS    DATASET2START
public static void main10(String[] args) {
    int num1 = 5;
    int num2 = 3;
    int num3 = 10;
    if (num1 > num2 && num1 > num3)
        System.out.println(num1);
    else if (num2 > num1 && num2 > num3)
        System.out.println(num2);
    else if (num3 > num1 && num3 > num2)
        System.out.println(num3);
}
}