# Data Structure



## Dictionary

### [defaultdict](https://www.geeksforgeeks.org/defaultdict-in-python/)

* **Using List as default_factory**

  ```python
  # Python program to demonstrate 
  # defaultdict 
  
  
  from collections import defaultdict 
  
  
  # Defining a dict 
  d = defaultdict(list) 
  
  for i in range(5): 
  	d[i].append(i) 
  	
  print("Dictionary with values as list:") 
  print(d) 
  mapping.values()
  ```

  ```
  Dictionary with values as list:
  defaultdict(<class 'list'>, {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]})
  ```

* **Using int as default_factory**

  ```python
  # Python program to demonstrate 
  # defaultdict 
  
  
  from collections import defaultdict 
  
  
  # Defining the dict 
  d = defaultdict(int) 
  
  L = [1, 2, 3, 4, 2, 4, 1, 2] 
  
  # Iterate through the list 
  # for keeping the count 
  for i in L: 
  	
  	# The default value is 0 
  	# so there is no need to 
  	# enter the key first 
  	d[i] += 1
  	
  print(d) 
  mapping.values()
  ```



### [OrderedDict](https://www.geeksforgeeks.org/ordereddict-in-python/)

* **Key value Change:** If the value of a certain key is changed, the position of the key remains unchanged in OrderedDict.
* **Deletion and Re-Inserting**: Deleting and re-inserting the same key will push it to the back as OrderedDict however maintains the order of insertion.

- `popitem`(*last=True*)

  The [`popitem()`](https://docs.python.org/3/library/collections.html#collections.OrderedDict.popitem) method for ordered dictionaries returns and removes a (key, value) pair. The pairs are returned in LIFO order if *last* is true or FIFO order if false.

- `move_to_end`(*key*, *last=True*)

  Move an existing *key* to either end of an ordered dictionary. The item is moved to the right end if *last* is true (the default) or to the beginning if *last* is false. Raises [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError) if the *key* does not exist:

  ```python
  from collections import OrderedDict 
  >>> d = OrderedDict.fromkeys('abcde')
  >>> d.move_to_end('b')
  >>> ''.join(d.keys())
  'acdeb'
  >>> d.move_to_end('b', last=False)
  >>> ''.join(d.keys())
  'bacde'
  ```

  

# Data Types



## Compare float with None: `float('inf')`

 ` defaultdict(<class 'int'>, {1: 2, 2: 3, 3: 1, 4: 2})`

