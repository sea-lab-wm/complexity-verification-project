//Look into this: using System;
import java.util.Random;
import java.io.*;
import java.util.*;
public class PasswordgeneratorSemanticNormal
{
    // PasswordGenerator: Generates a random password of the given strength
    // strength: length of the string to generate
    // random: random number generator (System.Random)
    // letters: letters
    // numbers: numbers
    // alphabet: alphabet
    // passphrase: passphrase
    // index: index
    // randomIndex: randomIndex
    // character: character

    public static char[] PasswordGenerator(int strength)
    {
        String letters = "abcdefghijklmnopqrstuvwxyz";
        String numbers = "0123456789";      //THESE 3 USED TO BE const
        String alphabet = letters + numbers;
	Random rand = new Random();
        char[] passphrase = new char[strength];
        for (int index = 0; index < strength; index++)
        {
            int randomIndex = rand.nextInt(alphabet.length()-1)+1;
            char character = alphabet.charAt(index);
            passphrase[index] = character;
        }
        return passphrase;
    }
}
