// Auto-generated from Jekyll posts
// Run 'node parse-posts.js' to regenerate

export interface Post {
  slug: string;
  date: string;
  dateObj: string;
  title: string;
  excerpt: string;
  content: string;
  tags: string[];
}

export const posts: Post[] = [
  {
    "slug": "loop-unwound",
    "date": "2025 · 09",
    "dateObj": "2025-09-09T06:00:00.000Z",
    "title": "The Loop Unwound - How Deep Networks Are Just Optimized Algorithms in Disguise",
    "excerpt": "The Loop Unwound: How Deep Networks Are Just Optimized Algorithms in Disguise Hey there, fellow travelers on the AI frontier! As a research engineer, especially one with a physics background, you've probably felt that familiar pull towards understanding the \"why\" behind the \"what works.\" We've seen ...",
    "content": "## The Loop Unwound: How Deep Networks Are Just Optimized Algorithms in Disguise\n\nHey there, fellow travelers on the AI frontier! As a research engineer, especially one with a physics background, you've probably felt that familiar pull towards understanding the \"why\" behind the \"what works.\" We've seen the incredible empirical success of Deep Neural Networks (DNNs) – ResNets, CNNs, Transformers – reshaping virtually every field. But have you ever paused to wonder if there's a deeper, unifying principle at play beyond just \"more data and bigger models\"?\n\nWhat if I told you that these architectural marvels, the ones we painstakingly design and train, can be viewed not as generic function approximators, but as **unrolled optimization algorithms**? And what if their goal, fundamentally, is to achieve better **compression** of data, in an information-theoretic sense?\n\nSounds a bit wild, right? But stick with me, because this concept, championed by folks like the MA-Lab at Berkeley, offers an incredibly elegant and powerful lens through which to understand, and even design, deep networks.\n\n### Learning as Compression: The Information-Theoretic Bedrock\n\nAs physicists, we instinctively appreciate parsimony. The universe often operates on elegant, minimal principles. In information theory, this translates to **compression**. If you can compress data, you've essentially captured its underlying structure and eliminated redundancy.\n\nThink about it:\n* **Dimensionality Reduction** (like PCA): It's literally about finding a lower-dimensional (compressed) representation.\n* **Generative Models** (like VAEs or GANs): They learn the latent, compressed manifold from which data samples can be generated.\n\nThe audacious claim here is that all of machine learning, at its core, is a quest for better data compression. And our beloved DNNs are the vehicles.\n\n### The \"Unrolling\" Revelation: From Iteration to Layers\n\nThis is where the magic happens, and where our optimization background comes in handy.\n\nRemember iterative algorithms? Like gradient descent, or the Iterative Soft-Thresholding Algorithm (ISTA) we often use for sparse recovery? They solve an optimization problem by repeatedly applying a set of rules until they converge to a solution.\n\n```python\n# Pseudo-code for a classic iterative algorithm\ncurrent_solution = initial_guess\nfor step in range(num_iterations):\n    current_solution = update_rule(current_solution, fixed_parameters)\nreturn current_solution\n````\n\nNow, imagine taking that `for` loop and literally \"unwinding\" it. Instead of a loop, we create a fixed sequence of operations, where each step of the original algorithm becomes a distinct \"layer\" in a neural network.\n\n```python\n# Pseudo-code for an unrolled network\nlayer_1_output = Layer1(initial_guess, learnable_parameters_1)\nlayer_2_output = Layer2(layer_1_output, learnable_parameters_2)\n...\nfinal_output = LayerN(layer_N_minus_1_output, learnable_parameters_N)\nreturn final_output\n```\n\n**Here's the kicker:** In this unrolled network, the `fixed_parameters` of the original algorithm (like the step size, or the regularization strength) are no longer fixed. They become **learnable weights** within each layer. And critically, each layer can have *its own set* of these learnable parameters.\n\n#### A Concrete Example: Denoising with Unrolled ISTA\n\nLet's ground this in something tangible. Consider the problem of recovering a sparse signal from a noisy one. The ISTA algorithm is a classic way to do this. It involves:\n\n1.  A **gradient step** (moving towards the noisy data).\n2.  A **proximal step** (like soft-thresholding, encouraging sparsity).\n\nIf we unroll ISTA:\n\n  * Each ResNet-like block in our \"Unrolled ISTA Network\" performs one iteration of these two steps.\n  * The soft-thresholding parameter (our $\\\\lambda$) is no longer a fixed hyperparameter we tune; it becomes a **learnable weight** within each layer.\n  * The \"gradient step\" is implicitly learned by the linear transformations and activations within the layer.\n\nWhen we train this unrolled network on noisy-to-clean data pairs, it learns the optimal parameters for each step of the denoising process. The magic? It often achieves better results in *far fewer layers* (iterations) than the original, hand-tuned ISTA algorithm would need. Why? Because the network has learned the most efficient path through the optimization landscape for the specific data it's seeing.\n\n#### Why This Is Incredible for Deep Learning\n\n1.  **Interpretability:** This isn't a black box. Each layer isn't just an arbitrary transformation; it's a mathematically grounded step towards solving a well-defined problem (e.g., iteratively removing noise, or finding a sparser representation).\n2.  **Principled Design:** Instead of guessing at architectures, we can start from robust, theoretically sound optimization algorithms. This gives us a blueprint for network design, potentially leading to simpler, more efficient architectures tailored for specific tasks.\n3.  **Efficiency:** Unrolled networks are often more efficient (fewer layers needed) because their parameters are learned directly from data, making them highly specialized and effective.\n4.  **Unifying Field Theory:** For us physics-minded folks, this is like finding a Grand Unified Theory for machine learning. It suggests that diverse architectures (ResNets, Transformers, etc.) might all be different \"flavors\" of unrolled optimization algorithms, each implicitly solving a compression problem in its own unique way.\n\n### The Future: Engineering Optimal Learning\n\nThis framework isn't just an academic curiosity; it has profound implications for how we'll engineer AI systems. Imagine designing a network for a specific inverse problem, like reconstructing an image from limited sensor data. Instead of throwing a generic CNN at it, we could start by unrolling a known, mathematically optimal algorithm for that inverse problem. The resulting network would inherit the theoretical guarantees of the algorithm while gaining the data-driven power of deep learning.\n\nIt's a testament to the beautiful convergence of classical mathematics and modern deep learning. The loop has been unwound, and in its layers, we find a clearer path to understanding, and perhaps even building, truly intelligent systems.\n\nWhat are your thoughts on this perspective? Have you encountered similar ideas in your work? Let's discuss in the comments below\\!\n\n```",
    "tags": []
  },
  {
    "slug": "multi-layer-architecture-node-js",
    "date": "2020 · 05",
    "dateObj": "2020-05-13T00:52:21.000Z",
    "title": "Multi Layer Architecture In Node.js",
    "excerpt": "We need a separation of concerns when building an API. Often times there will be a distinction between the , , and layer. The request layer is where we handle preparing the data to pass to the layer and also handling the responses. The layer is where all business logic is placed and calls to the dat...",
    "content": "We need a separation of concerns when building an API. Often times there will be a distinction between the `request`, `service`, and `repository` layer. \n\nThe request layer is where we handle preparing the data to pass to the `service` layer and also handling the responses. \n\nThe `service` layer is where all business logic is placed and calls to the database. We can handle all asyncronous tasks in this layer and will determine what request is appropriate to send back to client. \n\nThe `repository` layer is where the ORM works with the database. In our case we often use this to re-use common relational queries.",
    "tags": [
      "architecture",
      "node",
      "javascript"
    ]
  },
  {
    "slug": "using-jwt-for-access-tokens",
    "date": "2020 · 04",
    "dateObj": "2020-04-18T00:52:21.000Z",
    "title": "Using JWTs For API Access Tokens",
    "excerpt": "Introduction The old way of doing authentication is for you to manage an authentication server that issues credentials for your partners. An example of this would be something like IdentityServer4. However, this can easily be offloaded to a service such as Auth0 or Okta. A cloud provider makes the m...",
    "content": "# Introduction\n\nThe old way of doing authentication is for you to manage an authentication server that issues credentials for your partners. An example of this would be something like [IdentityServer4](https://identityserver4.readthedocs.io/en/latest/). However, this can easily be offloaded to a service such as Auth0 or Okta. A cloud provider makes the management and security much easier. \n\n<!-- We first need to understand OAuth2.0. Since the OAuth 2.0 standard does not specify which type of token is used. \n\nWe would like to authenticate API partners using OAuth2.0. However, the standard does not necessarily mean that we would use a JWT.  -->\n\n<!-- [auth0 on oauth2.0](https://auth0.com/docs/protocols/oauth2) -->\n\n\n# What is OAuth 2.0? \n\n[OAuth 2.0 is a protocol that allows a user to grant limited access to their resources on one site, to another site, without having to expose their credentials.](https://auth0.com/docs/protocols/oauth2)\n\nAs discussed in the Auth0 article, OAuth 2.0 has many different 'flavors' for obtaining a token for authentication. We are interested in a `machine-to-machine` grant that allows partners to access the protected resource (in this case our server).\n\nFor this we can use the [client credentials grant](https://oauth.net/2/grant-types/client-credentials/).\n\n# Why use JWTs For access? \n\nA partner is issued a key and secret key that they are then make a request to an authentication server (Auth0) that then issues them a JWT that they can use to make requests to our Node.js API. \n\nThe JWT is useful because it is asymetrically signed, it's opaque so a user can decode it and see what `scopes` and `audience` it has been issued for, and since it is self expiring it is less prone to abuse.  \n\n# How can we use OAuth 2.0 for authenticating our API users? \n\nInitially the OAuth language can be confusing and what relationships exist can be a little confusing. \nWe first create our API in Auth0. This is *our* protected resource, that we are merely registering in Auth0. We setup our scopes which should correspond to each of RESTful endpoints. For example, if we have a list users endpoint we should have a scope in Auth0 for `list:users`. \n\nNow, we create an `application` in Auth0. This is sometimes also called a `client` and you may see `client` in older docs from Auth0. \n\n\n# How to handle these JWTs in our Node API? \n\nThere are are some fantastic libraries which make our lives much easier. \n\nWe first need to authenticate the request. For this we need to consider is the JWT valid, and can use [authz]().\n\nNext we need to check the scopes of the authenticated request to see if the user has access to the endpoint they're requesting. For this we can use [](). \n\nNow, after all the middleware has ran we will have a `user` on the `req` object. That will have a `sub` claim which is essentially the unique `client_id`. Now we can use this `client_id` to look up our user and provide them with the data they're requesting. \n\n# Managing Auth0\n\nAuth0 of course has an SDK where we can manage the different applications, APIs, and everything else. \n\n[Auth0 SDK](https://www.npmjs.com/package/auth0)",
    "tags": [
      "authentication",
      "JSON",
      "Web"
    ]
  },
  {
    "slug": "fibonacci-sequence-in-javascript",
    "date": "2020 · 04",
    "dateObj": "2020-04-04T00:52:21.000Z",
    "title": "Fibonacci Sequence In JavaScript",
    "excerpt": "I was looking into the often posed question of how to solve the Fibonacci sequence in . I always figured if asked this I would just answer with \"Look up the list of the Fibonacci numbers and throw them in an array. Voila!\" However, it turns out there's perhaps more clever but less cheeky version whi...",
    "content": "I was looking into the often posed question of how to solve the Fibonacci sequence in `the language of your choice`. I always figured if asked this I would just answer with \"Look up the list of the Fibonacci numbers and throw them in an array. Voila!\" \n\n```javascript\nconst fibonacciNumbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811]\n\nfunction fibonacciMadman(n) {\n    return fibonacciNumbers[n];\n}\n\nconsole.log(fibonacciMadman(10));\n\\\\ 55\n```\n\nHowever, it turns out there's perhaps more clever but less cheeky version which is just to calculate them which I found in the comments. \n\n```javascript\nfunction fibonacci(n) {\n    return Math.round(\n        (Math.pow((1 + Math.sqrt(5)) / 2, n) - Math.pow(-2 / (1 + Math.sqrt(5)), n)) /\n        Math.sqrt(5)\n    );\n}\n\nconsole.log(fibonacci(22));\n\\\\ 55\n```\n\nI really enjoy thinking about 'old' problems in creative ways. Rather than just regurgitating what the CS Prof told us.",
    "tags": [
      "fibonacci",
      "sequence",
      "node"
    ]
  },
  {
    "slug": "running-postgresql-in-docker",
    "date": "2020 · 02",
    "dateObj": "2020-02-10T01:52:21.000Z",
    "title": "Running PostgreSQL in Docker",
    "excerpt": "PostgreSQL is like therapy after trying to wrangle Cassandra for years! Lets get it up and running in a docker container. I tried to install/run with brew but got an error connection refused. I think this has to do with other fuzion config found in /etc/hosts So instead we can just run it in a docke...",
    "content": "PostgreSQL is like therapy after trying to wrangle Cassandra for years! Lets get it up and running in a docker container.\n\nI tried to install/run with brew but got an error connection refused. I think this has to do with other fuzion config found in /etc/hosts \n\nSo instead we can just run it in a docker container. \n\n\n`$ docker run -d --name my_postgres -v my_dbdata:/var/lib/postgresql/data -p 54320:5432 postgres:11` \n\nCan check that it is running with the usual \n\n`$ docker ps -a` \n\n```bash\nCONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                      PORTS                                                       NAMES\n1c96b234b5ad        postgres:11                        \"docker-entrypoint.s…\"   4 minutes ago       Up 4 minutes                0.0.0.0:54320->5432/tcp                                     my_postgres\n3bc6faa05d0e        fuzion-kafka-docker_fuzion-kafka   \"start-kafka.sh\"         5 months ago        Exited (143) 2 months ago                                                               fuzion-kafka-docker_fuzion-kafka_1\n9321f95e9c08        wurstmeister/zookeeper             \"/bin/sh -c '/usr/sb…\"   5 months ago        Exited (137) 2 months ago                                                               fuzion-kafka-docker_fuzion-zookeeper_1\n4dddc20d66b7        cassandra:3.11                     \"docker-entrypoint.s…\"   19 months ago       Up 6 days                   7000-7001/tcp, 7199/tcp, 9160/tcp, 0.0.0.0:9042->9042/tcp   cassandra\n```\n\nCan view logs: \n\n`$ docker logs -f my_postgres`\n\n`$ docker exec -it my_postgres psql -U postgres`\n\nhttps://www.taniarascia.com/node-express-postgresql-heroku/\n\nFurther setup \n\nfor some reason when I do \n\n```sql\npostgres=# CREATE ROLE admin_user LOGIN PASSWORD 'johnnycat';\nCREATE ROLE\npostgres=# ALTER ROLE admin_user CREATEDB;\nALTER ROLE\npostgres=# \\q\n```\n\nI cannot do \n\n`$ docker exec -it my_postgres psql -U admin_user` \n\nMight have to tell which database to connect to. Obviously using 'postgres' user is not great but I'm just trying to get going. \n\nAnyways, we can create a database and then connect to it. \n\n`postgres=# CREATE DATABASE fuzion_files;`\n\n`postgres=# \\c fuzion_files;` \n\n\nRight off the bat: WE want to have UUIDs as the PK. \n\n`CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";`\n\n\n```sql\nCREATE TABLE meta_data (\n    file_id uuid DEFAULT uuid_generate_v4 (),\n    fuzion_event_id VARCHAR NOT NULL,\n    version VARCHAR NOT NULL,\n    name VARCHAR NOT NULL,\n    description VARCHAR,\n    type VARCHAR NOT NULL,\n    PRIMARY KEY (file_id)\n);\n```\n\n\n```sql\nINSERT INTO meta_data (\n    fuzion_event_id,\n    version,\n    name,\n    description\n)\nVALUES\n    (\n        '11E9F4595A14F3A0989B9BF9CE051B56',\n        '1',\n        'foo',\n        'a picture of a cat'\n    ),\n    (\n        '11E9F4C2518D7A908B85717A8E7E3383',\n        '12',\n        'bar',\n        'a pdf of cat pictures'\n    ),\n    (\n        '11EA3EDA364A8EF0A2F0E13FD80832FE',\n        '66',\n        'baz',\n        'pictures of chad in his comically low chair'\n    );\n```\n\nWe can now query our records very easily: \n\n`postgres=# select * from meta_data;` \n\n\nWe can connect to our postgres db in tableplus with \n\n---\nhost: 127.0.0.1 port:54320 \nuser: postgres\npassword: <blank> \nSSL: DISABLED \n---\n\nBOOOOOOMMMMMMMMMMM! \n\nI'm very excited to return back to the land of relational databases where we don't have to worry about querying by primary key like in Cassandra.",
    "tags": [
      "postgreSQL",
      "database",
      "docker"
    ]
  },
  {
    "slug": "serverless-data-lake-on-aws",
    "date": "2019 · 10",
    "dateObj": "2019-10-19T00:52:21.000Z",
    "title": "Serverless Data Lake On AWS",
    "excerpt": "What is a data lake and why is everyone talking about it?",
    "content": "What is a data lake and why is everyone talking about it?",
    "tags": [
      "aws",
      "spark",
      "s3"
    ]
  },
  {
    "slug": "introduction-to-pyspark",
    "date": "2019 · 04",
    "dateObj": "2019-04-22T00:52:21.000Z",
    "title": "Introduction to Pyspark",
    "excerpt": "I've spent a load of time learning about botht the internals of Spark as well as learning about Pyspark for analytics. I still need to collect my thoughts but this is more of a placeholder.",
    "content": "I've spent a load of time learning about botht the internals of Spark as well as learning about Pyspark for analytics. I still need to collect my thoughts but this is more of a placeholder.",
    "tags": [
      "spark",
      "pyspark",
      "big"
    ]
  },
  {
    "slug": "react-redux",
    "date": "2019 · 03",
    "dateObj": "2019-03-18T00:52:21.000Z",
    "title": "React and Redux - A New World",
    "excerpt": "Learning about components and managing state with redux. A great place to start it CRA",
    "content": "Learning about components and managing state with redux. \n\n[A great place to start it CRA](https://github.com/facebook/create-react-app)",
    "tags": [
      "JavaScript",
      "React",
      "Redux"
    ]
  },
  {
    "slug": "filtering-elements-by-value",
    "date": "2018 · 03",
    "dateObj": "2018-03-18T00:52:21.000Z",
    "title": "Filtering Elements By Value",
    "excerpt": "very good answer: https://stackoverflow.com/questions/286141/remove-blank-attributes-from-an-object-in-javascript Need to think about this one that uses recursion.",
    "content": "very good answer: https://stackoverflow.com/questions/286141/remove-blank-attributes-from-an-object-in-javascript \n\n```\nlet foo = {\n  bar: null,\n  baz: undefined,\n  dingetje: \"\",\n  good: \"cat\"\n};\n\nlet filter = key => {\n  return (foo[key] === null || foo[key] === undefined || foo[key] === \"\") && delete foo[key];\n}\n\n\nObject.keys(foo).forEach(filter);\n\nconsole.log(foo);\n>> {good: \"cat\" }\n```\n\nNeed to think about this one that uses recursion. \n\n```\nconst removeEmpty = obj =>\n  Object.fromEntries(\n    Object.entries(obj)\n      .filter(([k, v]) => v != null)\n      .map(([k, v]) => (typeof v === \"object\" ? [k, removeEmpty(v)] : [k, v]))\n  );\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "git-merge-tricks",
    "date": "2018 · 03",
    "dateObj": "2018-03-05T01:52:21.000Z",
    "title": "Git Merge Tricks",
    "excerpt": "Sometimes you might just want to merge and accept the master branch (or whomevers) changes because you know they will overrule you no matter what. So this is a pretty slick Git trick. Git merge ours and theirs Try this: To accept theirs changes: To accept yours: Nice! ##",
    "content": "Sometimes you might just want to merge and accept the master branch (or whomevers) changes because you know they will overrule you no matter what. So this is a pretty slick Git trick. \n\n\n## Git merge ours and theirs\n\nTry this:\n\nTo accept theirs changes: `git merge --strategy-option theirs`\n\nTo accept yours: `git merge --strategy-option ours`\n\nNice! \n\n##",
    "tags": [
      "JavaScript,",
      "Git"
    ]
  },
  {
    "slug": "manual-aws-requests",
    "date": "2018 · 02",
    "dateObj": "2018-02-06T01:52:21.000Z",
    "title": "Manually Making Requests to S3 (don't)",
    "excerpt": "You should probably be a normal human being and use the SDK, but one day I was tasked with figuring this out. So I figured I should document it. we need to derive the signing key and could use Crypto.js And then we have to manually sign requests if we are making making direct HTTP/HTTPS requests. An...",
    "content": "You should probably be a normal human being and use the SDK, but one day I was tasked with figuring this out. So I figured I should document it. \n\nwe need to derive the signing key and could use Crypto.js \n\n```\n<script src=\"http://crypto-js.googlecode.com/svn/tags/3.0.2/build/rollups/hmac-sha256.js\"></script>\n<script src=\"http://crypto-js.googlecode.com/svn/tags/3.0.2/build/components/enc-base64-min.js\"></script>\n\n<script>\n  var hash = CryptoJS.HmacSHA256(\"Message\", \"secret\");\n  var hashInBase64 = CryptoJS.enc.Base64.stringify(hash);\n  document.write(hashInBase64);\n</script>\n```\n\nAnd then we have to manually sign requests if we are making making direct HTTP/HTTPS requests. \n\n```javascript\nvar crypto = require(\"crypto-js\");\n\nfunction getSignatureKey(Crypto, key, dateStamp, regionName, serviceName) {\n    var kDate = Crypto.HmacSHA256(dateStamp, \"AWS4\" + key);\n    var kRegion = Crypto.HmacSHA256(regionName, kDate);\n    var kService = Crypto.HmacSHA256(serviceName, kRegion);\n    var kSigning = Crypto.HmacSHA256(\"aws4_request\", kService);\n    return kSigning;\n}\n```\nAnd to make the date in ISO 8601 format. \n\n```\nvar current_timestamp = new Date();\n\npm.environment.set(\"current_timestamp\", current_timestamp.toISOString());\n``` \n\ncheck here: https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html",
    "tags": [
      "JavaScript,",
      "AWS,",
      "S3"
    ]
  },
  {
    "slug": "find-string-in-javascript-list",
    "date": "2018 · 01",
    "dateObj": "2018-01-23T01:52:21.000Z",
    "title": "Finding String In JS Array",
    "excerpt": "Pretty slick, for JS.",
    "content": "`https://www.tjvantoll.com/2013/03/14/better-ways-of-comparing-a-javascript-string-to-multiple-values/`\n\n```javascript \n\nvar dingetje = 'tiger'; \n\n['tiger', 'foo', 'bar'].includes(dingetje); \n\n>> true \n\n``` \n\nPretty slick, for JS.",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "getting-timezone-from-google",
    "date": "2018 · 01",
    "dateObj": "2018-01-17T01:52:21.000Z",
    "title": "Using Google Maps API to Return the Local Time ",
    "excerpt": "This is some half baked code that we copy/pasted the variables around. Essentially just a proof of concept. This just shows how we could get a timestamp and geolocation from the browser, send that to the Maps API and get back both the offset and daylight savings offset too. From the docs: The local ...",
    "content": "`https://developers.google.com/maps/documentation/timezone/intro#Responses`\n\nThis is some half baked code that we copy/pasted the variables around. Essentially just a proof of concept. \n\nThis just shows how we could get a timestamp and geolocation from the browser, send that to the Maps API and get back both the offset and daylight savings offset too. \n\n\n```javascript \n\nfunction successFunction(position) {\n    var lat = position.coords.latitude;\n    var long = position.coords.longitude;\n    console.log('Your latitude is :'+lat+' and longitude is '+long);\n}\n\n// get the longitude and latitude from browser. \nif (navigator.geolocation) {\n    navigator.geolocation.getCurrentPosition(successFunction, errorFunction);\n} else {\n    alert('It seems like Geolocation, which is required for this page, is not enabled in your browser. Please use a browser which supports it.');\n}\n\n// get a timestamp and covnert stupid JS millisecond to seconds. \nvar foo = Math.floor(Date.now() / 1000);\n\n// send this off to Maps API. \nfetch('https://maps.googleapis.com/maps/api/timezone/json?location=43.6779426,-116.35970530000002&timestamp=1522881075211&key=AIzaSyBQ4TieD5Zik7Axu1Rp8LsyY3ayPEhXVHA')\n  .then(function(response) {\n    return response.json();\n  })\n  .then(function(myJson) {\n    console.log(myJson);\n  });\n```\n\nFrom the docs: \nThe local time of a given location is the sum of the timestamp parameter, and the dstOffset and rawOffset fields from the result.\n\nWe could then either use an online UNIX timestamp converter or we could just write some other code to \nturn it into a timestamp object.",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "lazy-loading-images",
    "date": "2018 · 01",
    "dateObj": "2018-01-09T01:52:21.000Z",
    "title": "Lazy Loading Images",
    "excerpt": "This takes a placeholder image in src tag and then we replace it with the actual data-src.",
    "content": "This takes a placeholder image in src tag and then we replace it with the actual data-src.  \n\n```html \n\n<img data-src=\"https://assets.imgix.net/unsplash/jellyfish.jpg?w=800&h=400&fit=crop&crop=entropy\"\n          src=\"https://assets.imgix.net/unsplash/jellyfish.jpg?w=800&h=400&fit=crop&crop=entropy&px=16&blur=200&fm=webp\"\n>\n\n<script>\n    // Script goes just before </body>\n    // Script from https://varvy.com/pagespeed/defer-images.html\n    function init() {\n        var imgDefer = document.getElementsByTagName('img');\n        for (var i=0; i<imgDefer.length; i++) {\n            if(imgDefer[i].getAttribute('data-src')) {\n                imgDefer[i].setAttribute('src',imgDefer[i].getAttribute('data-src'));\n            }\n        }\n    }\n    window.onload = init;\n</script>\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "dictionary-to-run-function",
    "date": "2018 · 01",
    "dateObj": "2018-01-07T01:52:21.000Z",
    "title": "Dictionary To Run Function",
    "excerpt": "Getting a taste of the power of Functional Programming. A similiar way of doing it is like this:",
    "content": "Getting a taste of the power of Functional Programming.\n\n```javascript\nvar lookup = 'johnnyCat'; \nvar foo = {'johnnyCat': () => $('#questionAuthorship').slideDown('medium')};\n\n// slide down the jQuery\nfoo[lookup]();\n```\n\nA similiar way of doing it is like this: \n\n```javascript \nvar katana = {\n  isSharp: true,\n  use: function(){\n    this.isSharp = !this.isSharp;\n  }\n};\nkatana.use();\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "thought-on-programming",
    "date": "2017 · 12",
    "dateObj": "2017-12-18T01:52:21.000Z",
    "title": "Thoughts On Programming",
    "excerpt": "Interesting thoughts from /u/drawkbox that sort of remind me of the 'The Zen of Python'. --- \"Programmers with lots of hours of maintaining code eventually evolve to return early, sorting exit conditions at top and meat of the methods at the bottom. Same way you evolve out of one liners. Same way co...",
    "content": "Interesting thoughts from /u/drawkbox that sort of remind me of the 'The Zen of Python'.\n\n\n---\n\"Programmers with lots of hours of maintaining code eventually evolve to return early, sorting exit conditions at top and meat of the methods at the bottom.\n\nSame way you evolve out of one liners.\n\nSame way comments are extra weight that should only be in public or algorithm/need to know areas.\n\nSame way braces go on the end of the method/class name to reduce LOC.\n\nSame way you move on from heavy OO to dicts/lists.\n\nSame way you go more composition instead of inheritance.\n\nSame way while/do/while usually fades away, and if needed exit conditions.\n\nSame way you move on from single condition bracket-less ifs. (debatable but more merge friendly and OP hasn't yet)\n\nSame way you get joy deleting large swaths of code.\n\nand many others on and on.\n\nUsually these come from hours of writing/maintaining code and styles that lead to bugs.\"\n\n---",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "async-await-ajax",
    "date": "2017 · 12",
    "dateObj": "2017-12-06T01:52:21.000Z",
    "title": "Async Await an AJAX Request",
    "excerpt": "This will only work on more modern browsers! :)",
    "content": "This will only work on more modern browsers! :) \n\n```javascript\n\nasync function doAjax(args) {\n    let result;\n\n    try {\n        result = await $.ajax({\n            url: ajaxurl,\n            type: 'POST',\n            data: args\n        });\n\n        return result;\n    } catch (error) {\n        console.error(error);\n    }\n}\n\n// two ways to call this async function: \n\n// Elsewhere in code, inside an async function\nconst stuff = await doAjax();\n\n\n\ndoAjax().then( (data) => doStuff(data) )\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "clever-jquery-chaining",
    "date": "2017 · 12",
    "dateObj": "2017-12-06T01:52:21.000Z",
    "title": "Clever jQuery Chaining",
    "excerpt": "\"With jQuery, you can chain together actions/methods. Chaining allows us to run multiple jQuery methods (on the same element) within a single statement.\"",
    "content": "\"With jQuery, you can chain together actions/methods.\n\nChaining allows us to run multiple jQuery methods (on the same element) within a single statement.\"\n\n```css \n.focus {\n    border-color:red;\n}\n```\n\n\n```javascript\n\n  $(document).ready(function() {\n\n    $('input').blur(function() {\n        $('input').removeClass(\"focus\");\n      })\n      .focus(function() {\n        $(this).addClass(\"focus\")\n      });\n  });\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "callback-hell",
    "date": "2017 · 12",
    "dateObj": "2017-12-04T01:52:21.000Z",
    "title": "Callback Hell",
    "excerpt": "Avoid the callback triangle of doom with three simple rules: 1. keep your code shallow 2. modularize 3. handle every single error example:",
    "content": "Avoid the callback triangle of doom with three simple rules: \n\n1. keep your code shallow \n2. modularize \n3. handle every single error \n\nexample: \n```javascript \n var fs = require('fs')\n\n fs.readFile('/Does/not/exist', handleFile)\n\n function handleFile (error, file) {\n   if (error) return console.error('Uhoh, there was an error', error)\n   // otherwise, continue on and use `file` in your code\n }\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "checking-empty-object",
    "date": "2017 · 11",
    "dateObj": "2017-11-16T01:52:21.000Z",
    "title": "Check For An Key In Object in JS and PHP",
    "excerpt": "Coming From Python this felt a little funny since Python has so many nice features for looking up keys and setting defaults etc. Check if object has a key in JavaScript https://stackoverflow.com/questions/455338/how-do-i-check-if-an-object-has-a-key-in-javascript check if array has a key in PHP",
    "content": "Coming From Python this felt a little funny since Python has so many nice features for looking up keys and setting defaults etc. \n\n## Check if object has a key in JavaScript\n\nhttps://stackoverflow.com/questions/455338/how-do-i-check-if-an-object-has-a-key-in-javascript\n\n`myObj.hasOwnProperty('key');` \n\n## check if array has a key in PHP \n\n`array_key_exists('filename', $resourcePairedValue)`",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "ajax-url-argument",
    "date": "2017 · 11",
    "dateObj": "2017-11-15T01:52:21.000Z",
    "title": "AJAX URL Path Issue",
    "excerpt": "You need to be careful becasue if in your pathURL argument you have a relative path (something like ../../foo/bar) it will break when we use that prototype/ ajax call within a different file because the ajax call uses the path from the CURRENT PAGE you are on. So, it seems if you want to make the sa...",
    "content": "You need to be careful becasue if in your pathURL argument you have a relative path (something like ../../foo/bar) it will break when we use that prototype/ ajax call within a different file because the ajax call uses the path from the CURRENT PAGE you are on. So, it seems if you want to make the same ajax call on two different pages you would have to use an absolute path to the API endpoint otherwise it will be broken. You can make an absolute path my using /main/foo/bar.php. \n\nfrom: https://stackoverflow.com/questions/24627075/jquery-ajax-url-path-issue\n\n```javascript\n// current url: http://sample.com/users\n// ajax code load from users page\n\n$.ajax({\n   url: '/yourFile.php',\n   ...\n});\n\n// ajax url will be: http://sample.com/yourFile.php\n```\n\n```javascript \n// current url: http://sample.com/users\n// ajax code load from users page\n\n$.ajax({\n    url: 'yourFile.php',\n    ...\n});\n\n//...ajax url will be http://simple.com/users/yourFile.php\n```",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "composer-and-phantom",
    "date": "2017 · 11",
    "dateObj": "2017-11-11T01:52:21.000Z",
    "title": "Composer and phantomjs (for PDF and Excel reports on Analytics)",
    "excerpt": "Despite the directions in the google drive they don't actually work because wants a file which doesn't get created when we do: . All it does is creates a file. And then when we try to use composer to install all of our projects dependencies which it would usually get from our file (much like npm's f...",
    "content": "Despite the directions in the google drive they don't actually work because `composer` wants a `composer.json` file which doesn't get created when we do: \n`$ brew install composer`. All it does is creates a `composer.lock` file. And then when we try to use composer to install all of our projects dependencies which it would \nusually get from our `composer.json` file (much like npm's `package.json` file) it can't find any! Well, we can just create one and copy/paste our dependencies into it and then all will, once again, be right in the world. \n\nInstall composer with brew: \n`$ brew install composer` \nYou can check the install with: \n`$ composer -V` \n\ncreate a composer.json file within project directory ( ~/Sites/project/ ): \n`$ vim composer.json` \n\nI went into to github and copy/pasted the `composer.lock` contents into our new `composer.lock` file. \n(use `:wq` to exit vim. :D )\n\nNow, we can run `$ composer install` which will go through and install all the dependencies. \nNow we'll have a /bin/ directory (before we didnt). `cd` into it and then give phantomjs read/write priviliges with. \n`$ chmod 777 phantomjs` \n\ngo and get your reporting on.",
    "tags": [
      "JavaScript"
    ]
  },
  {
    "slug": "json-encode",
    "date": "2017 · 10",
    "dateObj": "2017-10-21T00:52:21.000Z",
    "title": "PHP JSON parsing problem",
    "excerpt": "**Problem**: json_encode() returns FALSE. This means that our crap is broken. Intially it looked like the depth could be set wrong, but that didn't solve the issue. Fix: Use json_last_error_msg() which will output why it is broken. We could write an entire error handler so we would know about these ...",
    "content": "**Problem**: json_encode() returns FALSE. This means that our crap is broken. \nIntially it looked like the depth could be set wrong, but that didn't solve the issue. \nFix: Use json_last_error_msg() which will output why it is broken. \nWe could write an entire error handler so we would know about these things \n\nSomething like this would be sweet: \n```php \n<?php\n    if (!function_exists('json_last_error_msg')) {\n        function json_last_error_msg() {\n            static $ERRORS = array(\n                JSON_ERROR_NONE => 'No error',\n                JSON_ERROR_DEPTH => 'Maximum stack depth exceeded',\n                JSON_ERROR_STATE_MISMATCH => 'State mismatch (invalid or malformed JSON)',\n                JSON_ERROR_CTRL_CHAR => 'Control character error, possibly incorrectly encoded',\n                JSON_ERROR_SYNTAX => 'Syntax error',\n                JSON_ERROR_UTF8 => 'Malformed UTF-8 characters, possibly incorrectly encoded'\n            );\n\n            $error = json_last_error();\n            return isset($ERRORS[$error]) ? $ERRORS[$error] : 'Unknown error';\n        }\n    }\n?>\n```\n\nAlso, in PHP 5.5.0+ we could just change our json_encode to: \n \n``` php \necho json_encode($foo, JSON_PARTIAL_OUTPUT_ON_ERROR | JSON_PRETTY_PRINT);\n```",
    "tags": [
      "PHP,",
      "API"
    ]
  },
  {
    "slug": "arrays-in-php-7",
    "date": "2017 · 10",
    "dateObj": "2017-10-21T00:52:21.000Z",
    "title": "Arrays in PHP 7",
    "excerpt": "Turns out it's not good to instantiate an array by using: . PHP does not appreciate or tolerate such things. So we have to change them to in order for things to work properly.",
    "content": "Turns out it's not good to instantiate an array by using: \n`$foo = '';`. PHP does not appreciate or tolerate such things. \nSo we have to change them to `$foo = array();` in order for things to work properly.",
    "tags": [
      "PHP,",
      "API"
    ]
  },
  {
    "slug": "how-to-train-a-tiger",
    "date": "2017 · 06",
    "dateObj": "2017-06-30T00:52:21.000Z",
    "title": "How to Train A Tiger",
    "excerpt": "Training a Tiger is no simple task. You first need salmon. In actuality you need a LOT of salmon.",
    "content": "Training a Tiger is no simple task. You first need salmon. In actuality you need a LOT of salmon.",
    "tags": [
      "training"
    ]
  },
  {
    "slug": "building-my-photography-website",
    "date": "2015 · 03",
    "dateObj": "2015-03-15T23:52:21.000Z",
    "title": "Building My Photography Website",
    "excerpt": "Back in 2015 I decided I would dedicate my spring break to building my first website. Being a photographer it had to be as current and modern as possible. However, web development has gotten incredibly complicated so I chose to instead spend a few bucks on a nice parallax single page website. It see...",
    "content": "![Home Page](https://lh3.googleusercontent.com/pgqrZpVzULSb5dDqGxL2jfKHbjJ2EMXZacOi6ZhTDEaW3dRcWB1Z8qlqy6DDmX9Nli_SRajnEsexOVrbrCzNK3qucTsLcrEPbNpry1j5o-wbRfuSLr1L8o30pHtl2Hd5mORz18Tm9PaWRcmwmJWuLc9Ekeniy6XYGh0_aj1-TX_uMm20JfpyK1rmPASkWLRh7saH_ZL9Nx-OfGD899Lb54jUpK_uS0OHgwoeamPhFtXyjnEA_QKSF-ccbX9Lbl4NoiXUDXNAK-fjlIsfPBq7kJdZI4EEzLJ9Bg4R5NmVOiWSlcbPzSuYYSLARGz1F0n_YOOouho_qSTDgVfFuYn81N7g153DdbUTeDOX30cIPMnBGcFQdRwUDMGIKM8vwgUlficdcJdbxxdBpm7Fym9bnm0J9oIEH2gjzzc5KVL74ID-dFUj6dGFpSA_H6c2DlmW4OUNPaRDroaiUH9Xhv3VZBJ7fOQhWRr-A6bWMqs2oaFv5d983JR3SKpambQOtj2CKI1My5JGkc62CP6_O6mrg_rWizcVWAflz8SYU1AtoVCbe4vYU0mFmxb_6gYxJCryud1mJ4mP1exn5SC3cjjV_fR8jm6TjVXSQasrRNrz5ZoJZzccsHHftw=w1534-h957-no)\n\nBack in 2015 I decided I would dedicate my spring break to building my first website. Being a photographer it had to be as\ncurrent and modern as possible. However, web development has gotten incredibly complicated so I chose to instead spend a few bucks on\na nice parallax single page website. It seems like it would be relatively straightforward to just paste in your own photos, but\nsince it was my first foray into this new world it still took me the entirety of my spring break to choose a template,\ngo through my portfolio and choose the photos, tweak the template to my liking, and finally deploy the website.\n\nAt the time I went with GoDaddy for my domain name `jminorphotography.com`. They were actually quite good for the year that I had it hosted there.\nAfter the first year I decided not to renew the site since it was around $100 for the year, and it didn't see much traffic at all since most people have gone to more photo-centric website anyways.\n\nWell, now with GitHub hosting available it has been resurrected. Although I'm currently facing issues with it loading the JavaScript correctly (surprise surprise), but hopefully I can get it sorted out.\nIn the meantime I've taken some screenshots of the exact same [GitHub repo(!)](https://github.com/johnnydevriese/jminorphotography) running locally to show it off.\n\n\nAbove is the landing page with a carousel of my three favorite images at the time, and here is the 'About Me' part of the page once you scroll down.\n![About Me](https://lh3.googleusercontent.com/Ywz4VnpcP6Bxk7QMClC_RhbmHdie-YaCqUuYLXPqWwauLHELnIIDLLTmp3Z5M73oWVlDoYs_YgTHhXzakrJTxNLdk9fyjXWFGlpYBOstWiwJcOhMqW0XhmqATo_xFi9GiKdjGRQv4SxrYcuyzJl4mKZI-cLQc5r0leEFra8yy84gWarvNmFG-JrLDZhmsXOrWuk99rDAUGFfpIp0TpUeMlQHzTBKUO7KpmxYCeGBv6toy3gbwQswHFroQOWs9zpFdvQQhG18C9ITR4s4KsSsUX5yuyYCiMTsCDfMXil7jMJk0vgbT5B8nPTLBzw8q0xBOgglpfK0KthqOBBDgLvSTrwzxdnPLXHb2GvxF4Xst9y3iq_C47eZW3a9fQvPifSvaHcSG6ycNA507sjK9ivP0_I_kY_3RX3vHzb6GZ98Yx7svpQpUBmSvymRBLLulJq-RUa6yvKpszcHBnvdkx5YSSetD-wAcE6PkvD-zYDbbP5yLZQjW9dZV0w-SK46mEEsQRV4Q0lHVw8RSHEalfJiSYHiY6YdrTgjSklqp51UgIUKLZuf9j3fsBJatVTDr2oX2MBLaujXZ76VyHYJQbRQj9NrgriPrurQU4qxLWRUquaON1zTAQ=w1534-h957-no)\n\n\n\n<!--\n\n[Landing Page][website1]\n\n[About me][website2]\n\n[jared-github]: http://jaurentz.github.io/\n\n\n[website1]: https://lh3.googleusercontent.com/bGm_C0Esk7boBLFLs0MyhVCOeSulH2LOHbvePT_Bnd7DBBpK7SV0uCIq5sPVGtcIbFVbezIyVeK1lJE8SeZ65mcpDSNZ1kMv_TpDJuaqNjR3F8--KmX3FNUCjS-acSObpdd8tOI6P0I1pJZKnErtNmbNzQxO43Iq_Y6-wKVU7QVYvq0SquRlod2zQ3VYqriOscL0dcCUriMOnkJEmeglHjlQN6E4_EIor0ENNjWNztyb6phd8HBVNA2EzIBguzpapF9ig1G5alnIckd2o2FsYJ_vevLPNq1ZzSBY23MnVWJzS0f8abz2a6R_waMxqiCTyKJJuKt3sCKxWjcsKIvEkJgWGiF5X8o3mbLFFDlDu-J_jGGdmDa_6nZrFAEBSq-ZcggpIxbcLuqFtPOHk421Otehshh-0qRc8r1QBe2drwcHpYMsjqUBvAntyZ9LYBIgirjgdbfdoDb1PUbdL4rSBigN_XUWZRuqorC4vvAPTjNktX_FlWySq79Tl20VL_VQQ209h0CelPfXXZpVIR_W1HfYNBIyyONOwB1GMtxlEROCtxsU-ld2mhvKTbLlM5GnjrL9J8XQVrPNCRT_-_C5krOZY5JmThE-gYCSQdKn-fYU8mBsaQ=w339-h211-no\n[website2]: https://lh3.googleusercontent.com/Ywz4VnpcP6Bxk7QMClC_RhbmHdie-YaCqUuYLXPqWwauLHELnIIDLLTmp3Z5M73oWVlDoYs_YgTHhXzakrJTxNLdk9fyjXWFGlpYBOstWiwJcOhMqW0XhmqATo_xFi9GiKdjGRQv4SxrYcuyzJl4mKZI-cLQc5r0leEFra8yy84gWarvNmFG-JrLDZhmsXOrWuk99rDAUGFfpIp0TpUeMlQHzTBKUO7KpmxYCeGBv6toy3gbwQswHFroQOWs9zpFdvQQhG18C9ITR4s4KsSsUX5yuyYCiMTsCDfMXil7jMJk0vgbT5B8nPTLBzw8q0xBOgglpfK0KthqOBBDgLvSTrwzxdnPLXHb2GvxF4Xst9y3iq_C47eZW3a9fQvPifSvaHcSG6ycNA507sjK9ivP0_I_kY_3RX3vHzb6GZ98Yx7svpQpUBmSvymRBLLulJq-RUa6yvKpszcHBnvdkx5YSSetD-wAcE6PkvD-zYDbbP5yLZQjW9dZV0w-SK46mEEsQRV4Q0lHVw8RSHEalfJiSYHiY6YdrTgjSklqp51UgIUKLZuf9j3fsBJatVTDr2oX2MBLaujXZ76VyHYJQbRQj9NrgriPrurQU4qxLWRUquaON1zTAQ=w1534-h957-no\n-->",
    "tags": [
      "web",
      "development"
    ]
  },
  {
    "slug": "solving-the-schrodinger-equation-using-chebfun",
    "date": "2014 · 09",
    "dateObj": "2014-09-01T23:52:21.000Z",
    "title": "Solving The Schrodinger Equation Using Chebfun!",
    "excerpt": "My first introduction to scientific computing was working with [Jared L. Aurentz][jared-github]. Over the summer of 2014 we worked on developing an algorithm in C, Python, and eventually Matlab to solve the Schrodinger and then calculate the hyperpolerizatbility of with the given potential. You can ...",
    "content": "My first introduction to scientific computing was working with [Jared L. Aurentz][jared-github].\nOver the summer of 2014 we worked on developing an algorithm in C, Python, and eventually Matlab\nto solve the Schrodinger and then calculate the hyperpolerizatbility of with the given potential.\n\nYou can read through Jared's write brief write-up through Oxford's research group [Chebfun website][chebfun-writeup].\n\n\nI hope to put my significantly longer write up along with all of the code on GitHub soon!\n\n\n[chebfun-writeup]: http://www.chebfun.org/examples/ode-eig/OpticalResponse.html\n[jared-github]: http://jaurentz.github.io/\n\n\nHere's a plot from Wikipedia of the first five chebyshev polynomials.\n![Chebyshev Polynomials are fun](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chebyshev_Polynomials_of_the_First_Kind.svg/640px-Chebyshev_Polynomials_of_the_First_Kind.svg.png?1498838992669 \"Chebyshev polynomials are fun!\")\n\n<!--\nYou’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.\n\nTo add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.\n\nJekyll also offers powerful support for code snippets:\n\n{% highlight ruby %}\ndef print_hi(name)\n  puts \"Hi, #{name}\"\nend\nprint_hi('Tom')\n#=> prints 'Hi, Tom' to STDOUT.\n{% endhighlight %}\n\nCheck out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].\n\n[jekyll-docs]: http://jekyllrb.com/docs/home\n[jekyll-gh]:   https://github.com/jekyll/jekyll\n[jekyll-talk]: https://talk.jekyllrb.com/\n-->",
    "tags": [
      "chebfun"
    ]
  }
];

export function getPostBySlug(slug: string): Post | undefined {
  return posts.find(post => post.slug === slug);
}

export function getAllPosts(): Post[] {
  return posts;
}
