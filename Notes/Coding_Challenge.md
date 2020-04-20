# Coding Challenge via Python

## Factor of 3 and 5

Creat a filter which could return a list of numbers which has and only has two factors: 3 and 5. Like the following example: <img src="https://latex.codecogs.com/svg.latex?\Large&space;Number=3^x\times5^y" title="\Large Number = 3^x \times 5^y"/> Is what we are looking for.

```python
input = (1,75)
return = [15, 45, 75]
```

```python
def factor35(low,high):
    list = []
    for x in range(low,high+1):
        if x%3 ==0 and x%5 ==0:
            y = x
            while y%3 ==0:
                y=y/3
            while y%5 ==0:
                y=y/5
            if y ==1:
                list.append(x)
    return list
```

## Sock Merchant

Given an array of integers representing the color of each sock, determine how many pairs of socks with matching colors there are. For example, there are **n = 7** socks with colors with colors **ar = [1,2,1,2,1,3,2]**. There is one of pair **1** and one of color **2**. There are three odd socks left, one of each color. The number of pairs is **2**.

```python
def function(ar):
    unique = list(set(ar))
    sum = 0
    counts = [0] * len(unique)
    for item in ar:
        counts[unique.index(item)]+=1
    for num in counts:
        if num%2==0:
            sum+=num/2
        else:
            sum+=(num-1)/2
    return sum
```

## Jumping on the Clouds

For each game, Emma will get an array of clouds numbered **0** if they are safe or **1** if they must be avoided. For example, **c = [0,1,0,0,0,1,0]** indexed from **0** to **6**. The number on each cloud is its index in the list so she must avoid the clouds at indexes  **1** and **5**. 

```python
c  = [0,1,0,0,0,1,0]
output = 4
```

```python
def jumpingOnClouds(c):
    if len(c) == 1 : 
        return 0
    if len(c) == 2: 
        return 0 if c[1]==1 else 1
    if c[2]==1: 
        return 1 + jumpingOnClouds(c[1:])
    if c[2]==0:
        return 1 + jumpingOnClouds(c[2:])

```

```python
def jumpingOnClouds(c):
    length  = len(c)
    step = 0
    current = 0
    for i in range(0,length-1): 
        if i == current:
            if current == length -2:
                if c[current+1] == 0:
                    step += 1
            else:
                N = current + 2
                if c[N] == 0:
                    current = N
                    step += 1
                else:
                    current += 1
                    step += 1
    return step
```

## New Year's Chaos

It's New Year's Day and everyone's in line for the Wonderland rollercoaster ride! There are a number of people queued up, and each person wears a sticker indicating their initial position in the queue. Initial positions increment by  from  at the front of the line to  **n** at the back.

Any person in the queue can bribe the person directly in front of them to swap positions. If two people swap positions, they still wear the same sticker denoting their original places in line. One person can bribe at most two others. For example, if **n = 8** and **Person 5** bribes **Person 4** , the queue will look like this:**1,2,3,5,4,6,7,8**.

Fascinated by this chaotic queue, you decide you must know the minimum number of bribes that took place to get the queue into its current state!

**Function Description:**

Complete the function minimumBribes in the editor below. It must print an integer representing the minimum number of bribes necessary, or Too chaotic if the line configuration is not possible.
minimumBribes has the following parameter(s):
- q: an array of integers

```python
sample_input  = [2, 1, 5, 3, 4]
output = 3
sample_input  = [2, 5, 1, 3, 4]
output = 'Too chaotic'
```

```python
def minimumBribes(Q):
    #
    # initialize the number of moves
    moves = 0 
    #
    # decrease Q by 1 to make index-matching more intuitive
    # so that our values go from 0 to N-1, just like our
    # indices.  (Not necessary but makes it easier to
    # understand.)
    Q = [P-1 for P in Q]
    #
    # Loop through each person (P) in the queue (Q)
    for current_i,original_p in enumerate(Q):
        # i is the current position of P, while P is the
        # original position of P.
        #
        # First check if any P is more than two ahead of 
        # its original position
        if original_p - current_i > 2:
            print("Too chaotic")
            return
        #
        # From here on out, we don't care if P has moved
        # forwards, it is better to count how many times
        # P has RECEIVED a bribe, by looking at who is
        # ahead of P.  P's original position is the value of P.
        # Anyone who bribed P cannot get to higher than
        # one position in front if P's original position,
        # so we need to look from one position in front
        # of P's original position to one in front of P's
        # current position, and see how many of those 
        # positions in Q contain a number large than P.
        # In other words we will look from P-1 to i-1,
        # which in Python is range(P-1,i-1+1), or simply
        # range(P-1,i).  To make sure we don't try an
        # index less than zero, replace P-1 with
        # max(P-1,0)
        for item in range(max(original_p-1,0),current_i):
            if Q[item] > original_p:
                moves += 1
    print(moves)
```

## Minimum Swaps

You are given an unordered array consisting of consecutive integers [1, 2, 3, ..., n] without any duplicates. You are allowed to swap any two elements. You need to find the minimum number of swaps required to sort the array in ascending order.

```python	
def minimumSwaps(arr):
  
    ref_arr = sorted(arr)
    swaps = 0
    
    for i,v in enumerate(arr):
        correct_value = ref_arr[i]
        if v != correct_value:
            to_swap_ix = arr.index(correct_value)
            arr[to_swap_ix],arr[i] = arr[i], arr[to_swap_ix]
            swaps += 1
            
    return swaps
```

## Array Manipulation

Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value to each of the array element between two given indices, inclusive. Once all operations have been performed, return the maximum value in your array.

For example, the length of your array of zeros **n  = 10** . Your list of queries is as follows:

```
   a b k
   1 5 3
   4 8 7
   6 9 1
```

Add the values of **K** between  the indices **a** and **b** inclusive:

```
index->	 1 2 3  4  5 6 7 8 9 10
        [0,0,0, 0, 0,0,0,0,0, 0]
        [3,3,3, 3, 3,0,0,0,0, 0]
        [3,3,3,10,10,7,7,7,0, 0]
        [3,3,3,10,10,8,8,8,1, 0]
```

The larger value is **10** after all operations are performed.

The solution is like building a array the record the value change after this point:

```
input:
5 3
1 2 100
2 5 100
3 4 100

output:
200
```

```
The last row of the final matrix we have :
100 200 200 200 100

The soluation is like building a array the record the value change after this point:

300

200            ------------

100       ----              ----
 
  0   ----                      ----
  
 1st   100       -100    
 2nd         100             -100  
 3rd              100    -100
	0     1    2   3   4   5

```

```python
def arrayManipulation(n, queries):
    array = [0] * (n + 1)
    for query in queries: 
        a = query[0] - 1
        b = query[1]
        k = query[2]
        array[a] += k
        array[b] -= k   
    max_value = 0
    running_count = 0
    for i in array:
        running_count += i
        if running_count > max_value:
            max_value = running_count
```

## Sherlock and Anagrams

Two strings are [*anagrams*](http://en.wikipedia.org/wiki/Anagram) of each other if the letters of one string can be rearranged to form the other string. Given a string, find the number of pairs of substrings of the string that are anagrams of each other.

For example ***s = mom***, the list of all anagrammatic pairs is***[m, m],[mo, om]*** at positions***[[0],[2]],***

***[[0,1],[1,2] ]***respectively.

```python
def sherlockAndAnagrams(s):
    dict={}

    count=0

    for i in range(len(s)):

        for j in range(i+1,len(s)+1):

            list1= list(s[i:j].strip())

            list1.sort()

            transf=''.join(list1)

            if transf in dict: 

                count+=dict[transf]

                dict[transf]=dict[transf]+1

            else: dict[transf]=1  
    return count   
```

## Count Triplets

You are given an array and you need to find number of tripets***(i,j,k)*** of indices such that the elements at those indices are in [geometric progression](https://en.wikipedia.org/wiki/Geometric_progression) for a given common ratio ***r*** and . ***i < j < k***.

For example, . arr = ***[1,4,16,64]*** If , we have ***[1,4,16]*** and ***[4,16,64]*** at indices ***(0,1,2)*** and ***(1,2,3)***.

```python
def countTriplets(arr, r):
    if len(arr) <= 2:
        return 0
    map_arr = {}
    map_doubles = {}
    count = 0
    # Traversing the array from rear, helps avoid division
    for x in arr[::-1]:

        r_x = r*x
        r_r_x = r*r_x

        # case: x is the first element (x, x*r, x*r*r)
        count += map_doubles.get((r_x, r_r_x), 0)

        # case: x is the second element (x/r, x, x*r)
        map_doubles[(x,r_x)] = map_doubles.get((x,r_x), 0) + map_arr.get(r_x, 0)

        # case: x is the third element (x/(r*r), x/r, x)
        map_arr[x] = map_arr.get(x, 0) + 1
        
    return count
```

