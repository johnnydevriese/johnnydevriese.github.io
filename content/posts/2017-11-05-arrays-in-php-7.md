---
layout: post
title:  "Arrays in PHP 7"
date:   2017-10-20 18:52:21
categories: PHP, API 
---

Turns out it's not good to instantiate an array by using: 
`$foo = '';`. PHP does not appreciate or tolerate such things. 
So we have to change them to `$foo = array();` in order for things to work properly. 