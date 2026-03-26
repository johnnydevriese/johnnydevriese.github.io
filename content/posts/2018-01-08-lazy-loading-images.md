---
layout: post
title:  "Lazy Loading Images"
date:   2018-01-08 18:52:21
categories: JavaScript
---


This takes a placeholder image in src tag and then we replace it with the actual data-src.  

```html 

<img data-src="https://assets.imgix.net/unsplash/jellyfish.jpg?w=800&h=400&fit=crop&crop=entropy"
          src="https://assets.imgix.net/unsplash/jellyfish.jpg?w=800&h=400&fit=crop&crop=entropy&px=16&blur=200&fm=webp"
>

<script>
    // Script goes just before </body>
    // Script from https://varvy.com/pagespeed/defer-images.html
    function init() {
        var imgDefer = document.getElementsByTagName('img');
        for (var i=0; i<imgDefer.length; i++) {
            if(imgDefer[i].getAttribute('data-src')) {
                imgDefer[i].setAttribute('src',imgDefer[i].getAttribute('data-src'));
            }
        }
    }
    window.onload = init;
</script>
``` 