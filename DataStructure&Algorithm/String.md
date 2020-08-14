# Problems Solving Skills

### Pattern Searching



#### :star:Rabin-Karp Algorithm

```python
Input:  txt[] = "THIS IS A TEST TEXT"
        pat[] = "TEST"
Output: Pattern found at index 10
```

Like the Naive Algorithm, Rabin-Karp algorithm also slides the pattern one by one. But unlike the Naive algorithm, Rabin Karp algorithm matches the hash value of the pattern with the hash value of current substring of text, and if the hash values match then only it starts matching individual characters. So Rabin Karp algorithm needs to calculate hash values for following strings.

1) Pattern itself.
2) All the substrings of text of length m.

Since we need to efficiently calculate hash values for all the substrings of size m of text, we must have a hash function which has following property.
Hash at the next shift must be efficiently computable from the current hash value and next character in text or we can say *hash(txt[s+1 .. s+m])* must be efficiently computable from *hash(txt[s .. s+m-1])* and *txt[s+m]* i.e., *hash(txt[s+1 .. s+m])*= *rehash(txt[s+m], hash(txt[s .. s+m-1]))* and rehash must be O(1) operation.

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0

        if len(haystack) < len(needle):
            return -1

        #init hash value
        needle_hashval = 0
        haystack_hashval = 0
        max_pow = pow(26,len(needle) - 1)

        for i in range(len(needle)):
            needle_hashval = needle_hashval * 26 + ord(needle[i])
        for i in range(len(needle) - 1):
            haystack_hashval = haystack_hashval * 26 + ord(haystack[i])

        for i in range(len(haystack) - len(needle) + 1):

            haystack_hashval = haystack_hashval * 26 + ord(haystack[i + len(needle) - 1])
            if needle_hashval == haystack_hashval:
                if haystack[i:i + len(needle)] == needle:
                    return i

            haystack_hashval = haystack_hashval - ord(haystack[i])*max_pow

        return -1
```

#### :star:KMP Algorithm

Pattern searching is an important problem in computer science. When we do search for a string in notepad/word file or browser or database, pattern searching algorithms are used to show the search results.

We have discussed Naive pattern searching algorithm in the [previous post](https://www.geeksforgeeks.org/searching-for-patterns-set-1-naive-pattern-searching/). The worst case complexity of the Naive algorithm is O(m(n-m+1)). The time complexity of KMP algorithm is O(n) in the worst case.

The KMP matching algorithm uses degenerating property (pattern having same sub-patterns appearing more than once in the pattern) of the pattern and improves the worst case complexity to O(n). The basic idea behind KMP’s algorithm is: whenever we detect a mismatch (after some matches), we already know some of the characters in the text of the next window. We take advantage of this information to avoid matching the characters that we know will anyway match. Let us consider below example to understand this.

```cmd
Matching Overview
txt = "AAAAABAAABA" 
pat = "AAAA"
prefix = [0,1,2,0]

We compare first window of txt with pat
txt = "AAAAABAAABA" 
pat = "AAAA"  [Initial position]
We find a match. This is same as Naive String Matching.

In the next step, we compare next window of txt with pat.
txt = "AAAAABAAABA" 
pat =  "AAAA" [Pattern shifted one position]
This is where KMP does optimization over Naive. In this 
second window, we only compare fourth A of pattern
with fourth character of current window of text to decide 
whether current window matches or not. Since we know 
first three characters will anyway match, we skipped 
matching first three characters. 

Need of Preprocessing?
An important question arises from the above explanation, 
how to know how many characters to be skipped. To know this, 
we pre-process pattern and prepare an integer array 
lps[] that tells us the count of characters to be skipped.
```

```python
class Solution(object):
    def getPrefix(self,substr):
        prefix = [0 for _ in range(len(substr))]

        i = 1
        j = 0
        while i < len(substr)-1:
            print(i,j,substr[i],substr[j])
            if substr[i] == substr[j]:
                j += 1
                prefix[i] = j
                i += 1
            else:
                """
                when j is greater than 0, keep comparsion
                eg: aabaaac
                       aabaaac
                    i = 5, j = 2
                    substr[i] = 'a'
                    substr[j] = 'b'
                --> j = prefix[j-1] = 1
                --> substr[j] = 'a'
                """
                if j != 0:
                    j = prefix[j-1]
                else:
                    prefix[i] = 0
                    i += 1

        return prefix

    def strStr(self, haystack, needle):
        if not needle:
            return 0
        prefix = self.getPrefix(needle)
        j = 0
        for i, char in enumerate(haystack):
            print(i,j,needle[j],char)
            while j > 0 and needle[j] != char:
                # if not match, skip the previous prefix[j-1] elements
                j = prefix[j-1]
            if needle[j] == char:
                j += 1
            if j == len(needle):
                return i - j + 1
        return -1
```



### Store string with Hashing map

```python
def hashLetter(str):
    record = [0]*26
    for char in str:
        record[ord(char)-ord('a')]+=1
    return tuple(record)

```

```python
#use Counter to implement same funtionality
import collections
count = collections.Counter(s)
```



### Remove unwanted item in string

The `lstrip()` method removes any leading characters (space is the default leading character to remove)

```python
txt = ",,,,,ssaaww.....banana"

x = txt.lstrip(",.asw")

print(x)

result: 'banana'
```



### Python String isdigit() and its application

In Python, `isdigit()` is a built-in method used for string handling.
The isdigit() methods returns “True” if all characters in the string are digits, Otherwise, It returns “False”.
This function is used to check if the argument contains digits such as : **0123456789**

```python
Input : string = '15460'
Output : True

Input : string = '154ayush60'
Output : False
```



# Questions

