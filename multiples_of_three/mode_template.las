% % Only one class hold
-even_1(X,Y) :- invented_predicate(X,Y).
-invented_predicate(X,Y) :- even_1(X,Y).
-odd_1(X,Y) :- invented_predicate(X,Y).
-invented_predicate(X,Y) :- odd_1(X,Y).

#modeb(2,div_2(var(digit))).
#modeb(2,div_3(var(digit))).
#modeb(2,div_4(var(digit))).
#modeb(2,div_5(var(digit))).
#modeh(1,invented_predicate(var(digit),var(digit)), (anti_reflexive)).
3 ~ invented_predicate(V1,V2) :- div_2(V1), div_2(V2).
3 ~ invented_predicate(V1,V2) :- div_3(V2), div_3(V1).
3 ~ invented_predicate(V1,V2) :- div_4(V1), div_4(V2).
3 ~ invented_predicate(V1,V2) :- div_5(V1), div_5(V2).

#defined invented_predicate/2.