## Longest Common Substring

A string is said to be a child of a another string if it can be formed by deleting 0 or more characters from the other string. Given two strings of equal length, what's the longest string that can be constructed such that it is a child of both?

For example, ABCD and ABDC have two children with maximum length 3, ABC and ABD. They can be formed by eliminating either the D or C from both strings. Note that we will not consider ABCD as a common child because we can't rearrange characters and ABCD != ABDC.

#### Recursion:

For string s1 = 'azb', s2 = 'aab': 

```
lcs('azb','aab')
=('b' == 'b')=> 1+lcs('az','aa')
=('a' != 'z')=> 1 + max(lcs('a','aa'),lcs('az','a'))
```

```python
def commonChild(s1, s2):
    if s1 =='' or s2 =='':
        return 0
    if s1[-1] == s2[-1]:
        return 1+ commonChild(s1[0:-1],s2[0:-1])
    else:
			return max(commonChild(s1[0:-1],s2),commonChild(s1,s2[0:-1]))
```

#### DP:

|      |  ""  |  a   |  z   |  b   |
| :--: | :--: | :--: | :--: | :--: |
|  ""  |  0   |  0   |  0   |  0   |
|  a   |  0   |  1   |  1   |  1   |
|  a   |  0   |  1   |  1   |  1   |
|  b   |  0   |  1   |  1   |  2   |

```python
def commonChild(s1, s2):
    # row 0 = 0, column 0  = 0
    l1 = len(s1)
    l2 = len(s2)
    # we only need history of previous row
    lcs = [[0]*(len(s1)+1) for _ in range(2)]
    #lcs_letters = [['']*(len(s1)+1) for _ in range(2)]
    
    # i in s1 = i+1 in lcs
    for i in range(l1):
        # get index pointers for current and previous row
        current = (i+1)%2 
        previous = i%2 
        # j in s1 = j+1 in lcs
        for j in range(l2):
            # i and j are used to step forward in each string.
            # Now check if s1[i] and s2[j] are equal 
            if s1[i] == s2[j]:
                # Now we have found one longer sequence 
                # than what we had previously found.
                # so add 1 to the length of previous longest
                # sequence which we could have found at
                # earliest previous position of each string,
                # therefore subtract -1 from both i and j
                lcs[current][j+1] = (lcs[previous][j] + 1) 
                #lcs_letters[li1][j+1] = lcs_letters[li][j]+s1[li]

            # if not matching pair, then
            # get the biggest previous value
            elif lcs[current][j] > lcs[previous][j+1]:
                lcs[current][j+1] = lcs[current][j] 
                #lcs_letters[li1][j+1] = lcs_letters[li1][j]
            else:
                lcs[current][j+1] = lcs[previous][j+1] 
                #lcs_letters[li1][j+1] = lcs_letters[li][j+1]
    #print(lcs_letters[(i+1)%2][j+1])
    return lcs[(i+1)%2][j+1]
```

