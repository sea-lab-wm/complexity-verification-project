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

    public static int CountChildren(string people, int lower, int upper)
    {
        int children = 0;
        string[] numbers = people.Split(' ');
        for (int index = 0; index < numbers.Length; index++)
        {
            int personAge = int.Parse(numbers[children]);
            bool withinRange = (personAge >= lower && personAge <= upper);
            if (withinRange)
            {
                children += 1;
            }
        }
        return children;
    }
}