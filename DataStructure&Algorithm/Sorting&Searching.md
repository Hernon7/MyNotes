## Customize Functions

### Sorting

#### :star:Use [bisect](https://docs.python.org/3/library/bisect.html) helps sorting array

**Keep adding value into a sorted array**

```python
# input a sorted array
import bisect
def insort(arr,x):
    # Insort = insort_right, place item into sorted position  ---> much faster than sorting array yourself
    bisect.insort_right(arr,x)
    return arr
```

**Update value in a sorted array which has fixed lenght**

```python
import bisect
def pop_then_insort(arr, x, y):
    # Use bisect_left because item already exists in list, otherwise _right returns index+1
    idx = bisect.bisect_left(arr, x)
    # Remove existing item, pop should be faster than remove here
    arr.pop(idx)
    # Insort = insort_right, place item into sorted position  ---> much faster than sorting array yourself
    bisect.insort_right(rra, y)
    return arr
```

#### :star:Median

```python
# input a sorted array
def manual_median(arr):
    # Using built-in medians would sort the array themselves, that's too slow for us
    #arr.sort()
    num_items = len(arr)
    if num_items % 2 == 0:
        median = (arr[num_items//2] + arr[(num_items//2)-1])/2
    else:
        # You don't need to do -1 but I left it as a lesson
        median = arr[(num_items-1)//2]
    return median, arr
```

#### :star:Use `functools` to customize build-in functions​

> In Python, both `list.sort` method and `sorted` built-in function accepts an optional parameter named `key`, which is a function that, given an element from the list returns its sorting key.

```python
def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K
```

```python
from functools import cmp_to_key

def my_cmp(a, b):
    # some sorting comparison which is hard to express using a key function

class MyClass(cmp_to_key(my_cmp)):
    ...
```

```python
#Given a list of non negative integers, arrange them such that they form the largest number.
"""
Input: [3,30,34,5,9]
Output: "9534330"
"""
# If we use the default string comparator of sort(), and concatenate sorted strings,
# cases as ['3', '30'] will fail for '3' < '30' but we want '330' rather than '303'.
# If we use customized cmp_func such that string x is smaller than string y if x + y < y + x, '30' < '3', we will get '330' at last.
from functools import cmp_to_key
class Solution:        
    def largestNumber(self, nums):
        
        def cmp_func(x, y):
            """Sorted by value of concatenated string increasingly."""
            if x + y > y + x:
                return 1
            elif x == y:
                return 0
            else:
                return -1
            
        # Build nums contains all numbers in the String format.
        nums = [str(num) for num in nums]
        
        # Sort nums by cmp_func decreasingly.
        nums.sort(key = cmp_to_key(cmp_func), reverse = True)
        
        # Remove leading 0s, if empty return '0'.
        return ''.join(nums).lstrip('0') or '0'
```



