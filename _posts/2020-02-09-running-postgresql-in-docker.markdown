---
layout: post
title:  "Running PostgreSQL in Docker"
date:   2020-02-09 18:52:21
categories: postgreSQL database docker
---

PostgreSQL is like therapy after trying to wrangle Cassandra for years! Lets get it up and running in a docker container.

I tried to install/run with brew but got an error connection refused. I think this has to do with other fuzion config found in /etc/hosts 

So instead we can just run it in a docker container. 


`$ docker run -d --name my_postgres -v my_dbdata:/var/lib/postgresql/data -p 54320:5432 postgres:11` 

Can check that it is running with the usual 

`$ docker ps -a` 

```bash
CONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                      PORTS                                                       NAMES
1c96b234b5ad        postgres:11                        "docker-entrypoint.s…"   4 minutes ago       Up 4 minutes                0.0.0.0:54320->5432/tcp                                     my_postgres
3bc6faa05d0e        fuzion-kafka-docker_fuzion-kafka   "start-kafka.sh"         5 months ago        Exited (143) 2 months ago                                                               fuzion-kafka-docker_fuzion-kafka_1
9321f95e9c08        wurstmeister/zookeeper             "/bin/sh -c '/usr/sb…"   5 months ago        Exited (137) 2 months ago                                                               fuzion-kafka-docker_fuzion-zookeeper_1
4dddc20d66b7        cassandra:3.11                     "docker-entrypoint.s…"   19 months ago       Up 6 days                   7000-7001/tcp, 7199/tcp, 9160/tcp, 0.0.0.0:9042->9042/tcp   cassandra
```

Can view logs: 

`$ docker logs -f my_postgres`

`$ docker exec -it my_postgres psql -U postgres`

https://www.taniarascia.com/node-express-postgresql-heroku/

Further setup 

for some reason when I do 

```sql
postgres=# CREATE ROLE admin_user LOGIN PASSWORD 'johnnycat';
CREATE ROLE
postgres=# ALTER ROLE admin_user CREATEDB;
ALTER ROLE
postgres=# \q
```

I cannot do 

`$ docker exec -it my_postgres psql -U admin_user` 

Might have to tell which database to connect to. Obviously using 'postgres' user is not great but I'm just trying to get going. 

Anyways, we can create a database and then connect to it. 

`postgres=# CREATE DATABASE fuzion_files;`

`postgres=# \c fuzion_files;` 


Right off the bat: WE want to have UUIDs as the PK. 

`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`


```sql
CREATE TABLE meta_data (
    file_id uuid DEFAULT uuid_generate_v4 (),
    fuzion_event_id VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description VARCHAR,
    type VARCHAR NOT NULL,
    PRIMARY KEY (file_id)
);
```


```sql
INSERT INTO meta_data (
    fuzion_event_id,
    version,
    name,
    description
)
VALUES
    (
        '11E9F4595A14F3A0989B9BF9CE051B56',
        '1',
        'foo',
        'a picture of a cat'
    ),
    (
        '11E9F4C2518D7A908B85717A8E7E3383',
        '12',
        'bar',
        'a pdf of cat pictures'
    ),
    (
        '11EA3EDA364A8EF0A2F0E13FD80832FE',
        '66',
        'baz',
        'pictures of chad in his comically low chair'
    );
```

We can now query our records very easily: 

`postgres=# select * from meta_data;` 


We can connect to our postgres db in tableplus with 

---
host: 127.0.0.1 port:54320 
user: postgres
password: <blank> 
SSL: DISABLED 
---

BOOOOOOMMMMMMMMMMM! 

I'm very excited to return back to the land of relational databases where we don't have to worry about querying by primary key like in Cassandra.

