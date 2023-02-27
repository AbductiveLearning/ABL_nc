equal(X,Y) :- invented_predicate(X), invented_predicate(Y).

#modeb(1,less(var(box),var(box)), (anti_reflexive, positive)).
#modeb(1,zero(var(digit))).
#modeb(1,one(var(digit))).
#modeb(1,two(var(digit))).
#modeb(1,three(var(digit))).
#modeb(1,four(var(digit))).
#modeb(1,five(var(digit))).
#modeb(1,six(var(digit))).
#modeb(1,seven(var(digit))).
#modeb(1,eight(var(digit))).
#modeb(1,nine(var(digit))).
#modeb(1,invented_predicate(var(digit))).
#modeh(1,succ(var(digit),var(digit)), (anti_reflexive)).

#defined invented_predicate/1.