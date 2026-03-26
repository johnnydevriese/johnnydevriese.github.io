---
layout: post
title:  "Git Merge Tricks"
date:   2018-03-04 18:52:21
categories: JavaScript, Git
---
Sometimes you might just want to merge and accept the master branch (or whomevers) changes because you know they will overrule you no matter what. So this is a pretty slick Git trick. 


## Git merge ours and theirs

Try this:

To accept theirs changes: `git merge --strategy-option theirs`

To accept yours: `git merge --strategy-option ours`

Nice! 

## 