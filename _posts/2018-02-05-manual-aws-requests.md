---
layout: post
title:  "Manually Making Requests to S3 (don't)"
date:   2018-02-05 18:52:21
categories: JavaScript, AWS, S3
---

You should probably be a normal human being and use the SDK, but one day I was tasked with figuring this out. So I figured I should document it. 

we need to derive the signing key and could use Crypto.js 

```
<script src="http://crypto-js.googlecode.com/svn/tags/3.0.2/build/rollups/hmac-sha256.js"></script>
<script src="http://crypto-js.googlecode.com/svn/tags/3.0.2/build/components/enc-base64-min.js"></script>

<script>
  var hash = CryptoJS.HmacSHA256("Message", "secret");
  var hashInBase64 = CryptoJS.enc.Base64.stringify(hash);
  document.write(hashInBase64);
</script>
```

And then we have to manually sign requests if we are making making direct HTTP/HTTPS requests. 

```javascript
var crypto = require("crypto-js");

function getSignatureKey(Crypto, key, dateStamp, regionName, serviceName) {
    var kDate = Crypto.HmacSHA256(dateStamp, "AWS4" + key);
    var kRegion = Crypto.HmacSHA256(regionName, kDate);
    var kService = Crypto.HmacSHA256(serviceName, kRegion);
    var kSigning = Crypto.HmacSHA256("aws4_request", kService);
    return kSigning;
}
```
And to make the date in ISO 8601 format. 

```
var current_timestamp = new Date();

pm.environment.set("current_timestamp", current_timestamp.toISOString());
``` 

check here: https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html