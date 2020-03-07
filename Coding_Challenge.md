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


