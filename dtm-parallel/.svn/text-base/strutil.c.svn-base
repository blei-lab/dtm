// Copyright 2010 Sean Gerrish
// All Rights Reserved
//
// Author: Sean Gerrish
//
// Defines a number of string functions.

#include <vector.h>
#include <string.h>
#include <stdarg.h>

using namespace std;

string StringPrintf(const char* format, ...) {
  char result[4000];
  va_list arg;
  va_start(arg, format);
  vsprintf (result, format, arg );
  va_end(arg);

  return result;
}

// Split a string on some given delimiter.
void SplitStringUsing(const string& s1,
		      const char* delimiter,
		      vector<string>* result) {
  // First, count the number of parts after the split.
  int i=0;
  while (i < s1.size()) {
    size_t n = s1.find_first_of(delimiter, i);
    if (n == string::npos) {
      result->push_back(s1.substr(i, s1.length() - i));
      return;
    }
    result->push_back(s1.substr(i, n - i));

    i = n + 1;
  }
}

// Split a string on some given delimiter.
string JoinStringUsing(const vector<string>& parts,
		       const char* delimiter) {
  string result;
  // First, count the number of parts after the split.
  for (vector<string>::const_iterator it=parts.begin();
       it != parts.end();
       ++it) {
    if (it == parts.begin()) {
      result = *it;
    } else {
      result += "," + *it;
    }
  }
}
