---
layout: post
title:  "Using JWTs For API Access Tokens"
date:   2020-04-17 18:52:21
categories: authentication JSON Web Tokens node javascript
---

# Introduction

The old way of doing authentication is for you to manage an authentication server that issues credentials for your partners. An example of this would be something like [IdentityServer4](https://identityserver4.readthedocs.io/en/latest/). However, this can easily be offloaded to a service such as Auth0 or Okta. A cloud provider makes the management and security much easier. 

<!-- We first need to understand OAuth2.0. Since the OAuth 2.0 standard does not specify which type of token is used. 

We would like to authenticate API partners using OAuth2.0. However, the standard does not necessarily mean that we would use a JWT.  -->

<!-- [auth0 on oauth2.0](https://auth0.com/docs/protocols/oauth2) -->


# What is OAuth 2.0? 

[OAuth 2.0 is a protocol that allows a user to grant limited access to their resources on one site, to another site, without having to expose their credentials.](https://auth0.com/docs/protocols/oauth2)

As discussed in the Auth0 article, OAuth 2.0 has many different 'flavors' for obtaining a token for authentication. We are interested in a `machine-to-machine` grant that allows partners to access the protected resource (in this case our server).

For this we can use the [client credentials grant](https://oauth.net/2/grant-types/client-credentials/).

# Why use JWTs For access? 

A partner is issued a key and secret key that they are then make a request to an authentication server (Auth0) that then issues them a JWT that they can use to make requests to our Node.js API. 

The JWT is useful because it is asymetrically signed, it's opaque so a user can decode it and see what `scopes` and `audience` it has been issued for, and since it is self expiring it is less prone to abuse.  

# How can we use OAuth 2.0 for authenticating our API users? 

Initially the OAuth language can be confusing and what relationships exist can be a little confusing. 
We first create our API in Auth0. This is *our* protected resource, that we are merely registering in Auth0. We setup our scopes which should correspond to each of RESTful endpoints. For example, if we have a list users endpoint we should have a scope in Auth0 for `list:users`. 

Now, we create an `application` in Auth0. This is sometimes also called a `client` and you may see `client` in older docs from Auth0. 


# How to handle these JWTs in our Node API? 

There are are some fantastic libraries which make our lives much easier. 

We first need to authenticate the request. For this we need to consider is the JWT valid, and can use [authz]().

Next we need to check the scopes of the authenticated request to see if the user has access to the endpoint they're requesting. For this we can use [](). 

Now, after all the middleware has ran we will have a `user` on the `req` object. That will have a `sub` claim which is essentially the unique `client_id`. Now we can use this `client_id` to look up our user and provide them with the data they're requesting. 

# Managing Auth0

Auth0 of course has an SDK where we can manage the different applications, APIs, and everything else. 

[Auth0 SDK](https://www.npmjs.com/package/auth0)


