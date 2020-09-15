# Problems



## Longest Common Substring

A string is said to be a child of a another string if it can be formed by deleting 0 or more characters from the other string. Given two strings of equal length, what's the longest string that can be constructed such that it is a child of both?

For example, ABCD and ABDC have two children with maximum length 3, ABC and ABD. They can be formed by eliminating either the D or C from both strings. Note that we will not consider ABCD as a common child because we can't rearrange characters and ABCD != ABDC.

**Recursion:**

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

**DP:**

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



## [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:

```
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```



Example 2:

```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

**Solution 1 Brute Force**

Time: O(2^n)
Space: O(n)

```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0 or n == 1 or n == 2 or n == 3:
            return n
        else:
            return self.climbStairs(n-1)+self.climbStairs(n-2)
```

This solution will experience runtime error

**Solution 2 memoization**

```
class Solution:
    cache = {}    
    def climbStairs(self, n):
        if n < 3:
            return n
        else:
            return self._climbStairs(n-1) + self._climbStairs(n-2)
    def _climbStairs(self, n):
        if n not in self.cache.keys():
            self.cache[n] = self.climbStairs(n)
        return self.cache[n]
```

use memoization to reduce redundant processing

**Solution 3 Loop Fibonacci**

```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0 or n == 1 or n == 2:
            return n
        else:
            res, step_1, step_2 = 0,1,2
            for i in range(2, n):
                res = step_2 + step_1
                step_1 = step_2
                step_2 = res
            return res
```

Time: O(n)
Space: O(1)

Basically a for loop Fibonacci approach.



## [Maximum Subarrary](https://leetcode.com/problems/maximum-subarray/)

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return *its sum*.

**Example**

```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

**Sulution**

```python
class Solution:
    def maxContiguousSubarraySum(self, nums):
        '''
        :type nums: list of int
        :rtype: int
        '''
        
        """
        We default to say the the best maximum seen so far is the first
        element.

        We also default to say the the best max ending at the first element
        is...the first element.
        """
        max_so_far = nums[0]
        max_ending_here = nums[0]
            
        # We will investigate the rest of the items in the array from index 1 onward.
        for i in range(1, len(nums)):
            """
            We are inspecting the item at index i.
    
            We want to answer the question:
            "What is the Max Contiguous Subarray Sum we can achieve ending at index i?"
    
            We have 2 choices:
    
            maxEndingHere + nums[i] (extend the previous subarray best whatever it was)
              1.) Let the item we are sitting at contribute to this best max we achieved
              ending at index i - 1.
    
            nums[i] (start and end at this index)
              2.) Just take the item we are sitting at's value.
    
            The max of these 2 choices will be the best answer to the Max Contiguous
            Subarray Sum we can achieve for subarrays ending at index i.
    
            Example:
    
            index     0  1   2  3   4  5  6   7  8
            Input: [ -2, 1, -3, 4, -1, 2, 1, -5, 4 ]
                     -2, 1, -2, 4,  3, 5, 6,  1, 5    'maxEndingHere' at each point
            
            The best subarrays we would take if we took them:
              ending at index 0: [ -2 ]           (snippet from index 0 to index 0)
              ending at index 1: [ 1 ]            (snippet from index 1 to index 1) [we just took the item at index 1]
              ending at index 2: [ 1, -3 ]        (snippet from index 1 to index 2)
              ending at index 3: [ 4 ]            (snippet from index 3 to index 3) [we just took the item at index 3]
              ending at index 4: [ 4, -1 ]        (snippet from index 3 to index 4)
              ending at index 5: [ 4, -1, 2 ]     (snippet from index 3 to index 5)
              ending at index 6: [ 4, -1, 2, 1 ]  (snippet from index 3 to index 6)
              ending at index 7: [ 4, -1, 2, 1, -5 ]    (snippet from index 3 to index 7)
              ending at index 8: [ 4, -1, 2, 1, -5, 4 ] (snippet from index 3 to index 8)
    
            Notice how we are changing the end bound by 1 everytime.
            """
            max_ending_here = max(max_ending_here + nums[i], nums[i])
            
            # Did we beat the 'maxSoFar' with the 'maxEndingHere'?
            max_so_far = max(max_ending_here, max_so_far)

        return max_so_far		
```

