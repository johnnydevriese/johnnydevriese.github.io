---
layout: post
title:  "Filtering Elements By Value"
date:   2018-03-04 18:52:21
categories: JavaScript, Git
---


very good answer: https://stackoverflow.com/questions/286141/remove-blank-attributes-from-an-object-in-javascript 

```
let foo = {
  bar: null,
  baz: undefined,
  dingetje: "",
  good: "cat"
};

let filter = key => {
  return (foo[key] === null || foo[key] === undefined || foo[key] === "") && delete foo[key];
}


Object.keys(foo).forEach(filter);

console.log(foo);
>> {good: "cat" }
```

Need to think about this one that uses recursion. 

```
const removeEmpty = obj =>
  Object.fromEntries(
    Object.entries(obj)
      .filter(([k, v]) => v != null)
      .map(([k, v]) => (typeof v === "object" ? [k, removeEmpty(v)] : [k, v]))
  );
```