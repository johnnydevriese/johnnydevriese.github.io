---
layout: post
title:  "Dictionary To Runm Function"
date:   2017-12-17 18:52:21
categories: JavaScript
---

Getting a taste of the power of Functional Programming.

```javascript
var lookup = 'johnnyCat'; 
var foo = {'johnnyCat': () => $('#questionAuthorship').slideDown('medium')};

// slide down the jQuery
foo[lookup]();
```

A similiar way of doing it is like this: 

```javascript 
var katana = {
  isSharp: true,
  use: function(){
    this.isSharp = !this.isSharp;
  }
};
katana.use();
```