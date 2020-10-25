# Math



## Count Primes

Count the number of prime numbers less than a non-negative number, `n`.

```cmd
Input: n = 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.

Input: n = 1
Output: 0
```

The [Sieve of Eratosthenes](http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) is one of the most efficient ways to find all prime numbers up to *n*. But don't let that name scare you, I promise that the concept is surprisingly simple.

![https://leetcode.com/static/images/solutions/Sieve_of_Eratosthenes_animation.gif]()

4 is not a prime because it is divisible by 2, which means all multiples of 4 must also be divisible by 2 and were already marked off. So we can skip 4 immediately and go to the next number, 5. Now, all multiples of 5 such as 5 × 2 = 10, 5 × 3 = 15, 5 × 4 = 20, 5

In fact, we can mark off multiples of 5 starting at 5 × 5 = 25, because 5 × 2 = 10 was already marked off by multiple of 2, similarly 5 × 3 = 15 was already marked off by multiple of 3. Therefore, if the current number is *p*, we can always mark off

Yes, the terminating loop condition can be *p* < √*n*, as all non-primes ≥ √*n* must have already been marked off. When the loop terminates, all the numbers in the table that are non-marked are prime.

The Sieve of Eratosthenes uses an extra O(*n*) memory and its runtime complexity is O(*n* log log *n*). For the more mathematically inclined readers, you can read more about its algorithm complexity on [Wikipedia](http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Algorithm_complexity).

```java
public int countPrimes(int n) {
   boolean[] isPrime = new boolean[n];
   for (int i = 2; i < n; i++) {
      isPrime[i] = true;
   }
   // Loop's ending condition is i * i < n instead of i < sqrt(n)
   // to avoid repeatedly calling an expensive function sqrt().
   for (int i = 2; i * i < n; i++) {
      if (!isPrime[i]) continue;
      for (int j = i * i; j < n; j += i) {
         isPrime[j] = false;
      }
   }
   int count = 0;
   for (int i = 2; i < n; i++) {
      if (isPrime[i]) count++;
   }
```

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        """
        :type: n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        isPrime = [True for _ in range(n)]
        if n < 2:
            return 0
        for i in range(2):
            isPrime[i] = False
        for i in range(2,n):
            if isPrime[i]:
                j = i*i
                while j < n:
                    isPrime[j] = False
                    j += i
        return sum(isPrime)
```

