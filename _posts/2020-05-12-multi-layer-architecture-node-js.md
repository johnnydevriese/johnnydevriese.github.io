---
layout: post
title:  "Multi Layer Architecture In Node.js"
date:   2020-05-12 18:52:21
categories:  architecture node javascript
---

We need a separation of concerns when building an API. Often times there will be a distinction between the `request`, `service`, and `repository` layer. 

The request layer is where we handle preparing the data to pass to the `service` layer and also handling the responses. 

The `service` layer is where all business logic is placed and calls to the database. We can handle all asyncronous tasks in this layer and will determine what request is appropriate to send back to client. 

The `repository` layer is where the ORM works with the database. In our case we often use this to re-use common relational queries. 







