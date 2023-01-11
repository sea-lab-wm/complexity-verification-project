import java.util.*;
import java.io.*;
public class CountchildrenSemanticNormal
{
    // CountChildren: Returns the number of children within a list of people's ages
    // people: ages, separated by spaces
    // lower: lower inclusive boundary of ages to count
    // upper: upper inclusive boundary of ages to count
    // children: children
    // numbers: numbers
    // index: index
    // personAge: personAge
    // withinRange: withinRange

    public static int CountChildren(String people, int lower, int upper)
    {
        int children = 0;
        String[] numbers = people.trim().split("\\s+");
        for (int index = 0; index < numbers.length; index++)
        {
            int personAge = Integer.parseInt(numbers[children]);
            boolean withinRange = (personAge >= lower && personAge <= upper);
            if (withinRange)
            {
                children += 1;
            }
        }
        return children;
    }
}
