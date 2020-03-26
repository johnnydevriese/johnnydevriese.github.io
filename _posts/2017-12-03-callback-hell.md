---
layout: post
title:  "Callback Hell"
date:   2017-12-03 18:52:21
categories: JavaScript
---

Avoid the callback triangle of doom with three simple rules: 

1. keep your code shallow 
2. modularize 
3. handle every single error 

example: 
```javascript 
 var fs = require('fs')

 fs.readFile('/Does/not/exist', handleFile)

 function handleFile (error, file) {
   if (error) return console.error('Uhoh, there was an error', error)
   // otherwise, continue on and use `file` in your code
 }
```
