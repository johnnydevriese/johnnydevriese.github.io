---
layout: post
title:  "Dictionary To Run Function"
date:   2018-01-06 18:52:21
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