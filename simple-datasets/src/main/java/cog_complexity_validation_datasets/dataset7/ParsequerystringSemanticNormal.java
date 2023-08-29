//using System.Collections.Generic;
import java.util.HashMap;
import java.io.*;
import java.util.*;
public class ParsequerystringSemanticNormal
{
    // ParseQueryString: Parses a http query-string
    // querystring: raw query-string containing key=value pairs, separated by &
    // parts: parts
    // query: query
    // part: part
    // setting: setting
    // parameter: parameter
    // parameterValue: parameterValue

    public static HashMap<String, String> ParseQueryString(String querystring)
    {
        String[] parts = querystring.split("&", 2);
        HashMap<String, String> query = new HashMap<String, String>();

        for(String part : parts)
        {
            String[] setting = part.split("=",2);
            String parameter = setting[0];
            String parameterValue = parts[1];

            query.put(parameter, parameterValue);
        }
        return query;
    }
}
