---
layout: post
title:  "Using Google Maps API to Return the Local Time "
date:   2018-01-16 18:52:21
categories: JavaScript
---


`https://developers.google.com/maps/documentation/timezone/intro#Responses`

This is some half baked code that we copy/pasted the variables around. Essentially just a proof of concept. 

This just shows how we could get a timestamp and geolocation from the browser, send that to the Maps API and get back both the offset and daylight savings offset too. 


```javascript 

function successFunction(position) {
    var lat = position.coords.latitude;
    var long = position.coords.longitude;
    console.log('Your latitude is :'+lat+' and longitude is '+long);
}

// get the longitude and latitude from browser. 
if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(successFunction, errorFunction);
} else {
    alert('It seems like Geolocation, which is required for this page, is not enabled in your browser. Please use a browser which supports it.');
}

// get a timestamp and covnert stupid JS millisecond to seconds. 
var foo = Math.floor(Date.now() / 1000);

// send this off to Maps API. 
fetch('https://maps.googleapis.com/maps/api/timezone/json?location=43.6779426,-116.35970530000002&timestamp=1522881075211&key=AIzaSyBQ4TieD5Zik7Axu1Rp8LsyY3ayPEhXVHA')
  .then(function(response) {
    return response.json();
  })
  .then(function(myJson) {
    console.log(myJson);
  });
```

From the docs: 
The local time of a given location is the sum of the timestamp parameter, and the dstOffset and rawOffset fields from the result.

We could then either use an online UNIX timestamp converter or we could just write some other code to 
turn it into a timestamp object. 