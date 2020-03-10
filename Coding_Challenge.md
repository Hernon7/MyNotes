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

## Some

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

