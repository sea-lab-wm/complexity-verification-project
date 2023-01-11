//using System.Collections.Generic;
import java.util.*;
import java.io.*;
import java.util.HashMap;
public class ReadiniSemanticNormal
{
    // ReadIni: Parses the lines of an ini file
    // lines: collection of lines containing one setting each, like key=value
    // settings: settings
    // rawLine: rawLine
    // line: line
    // setting: setting
    // identifier: identifier
    // property: property

    public static HashMap<String, String> ReadIni(List<String> lines)
    {
        HashMap<String, String> settings = new HashMap<String, String>();
        for(String rawLine: lines)
        {
            String line = rawLine.trim();
            String[] setting = line.split("=",2);

            String identifier = setting[0];
            String property = setting[1];

            settings.put(identifier, line);
        }
        return settings;
    }
}
