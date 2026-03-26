---
layout: post
title:  "Fibonacci Sequence In JavaScript"
date:   2020-04-03 18:52:21
categories: fibonacci sequence node javascript
---

I was looking into the often posed question of how to solve the Fibonacci sequence in `the language of your choice`. I always figured if asked this I would just answer with "Look up the list of the Fibonacci numbers and throw them in an array. Voila!" 

```javascript
const fibonacciNumbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811]

function fibonacciMadman(n) {
    return fibonacciNumbers[n];
}

console.log(fibonacciMadman(10));
\\ 55
```

However, it turns out there's perhaps more clever but less cheeky version which is just to calculate them which I found in the comments. 

```javascript
function fibonacci(n) {
    return Math.round(
        (Math.pow((1 + Math.sqrt(5)) / 2, n) - Math.pow(-2 / (1 + Math.sqrt(5)), n)) /
        Math.sqrt(5)
    );
}

console.log(fibonacci(22));
\\ 55
```

I really enjoy thinking about 'old' problems in creative ways. Rather than just regurgitating what the CS Prof told us. 
