
attack((X1,Y1),Type1,(X2,Y2)) :- invented_predicate(Type1), invented_predicate_cond((X1,Y1),(X2,Y2)).

#modeb(1,left(var(pos),var(pos)), (anti_reflexive, positive)).
% #modeb(1,right(var(pos),var(pos)), (anti_reflexive, positive)).
#modeb(1,forward(var(pos),var(pos)), (anti_reflexive, positive)).
% #modeb(1,backward(var(pos),var(pos)), (anti_reflexive, positive)).
#modeb(1,left_forward(var(pos),var(pos)), (anti_reflexive, positive)).
% #modeb(1,left_backward(var(pos),var(pos)), (anti_reflexive, positive)).
#modeb(1,right_forward(var(pos),var(pos)), (anti_reflexive, positive)).
% #modeb(1,right_backward(var(pos),var(pos)), (anti_reflexive, positive)).
% #modeb(1,straight(var(pos),var(pos)), (anti_reflexive, symmetric, positive)).
% #modeb(1,diag(var(pos),var(pos)), (anti_reflexive, symmetric, positive)).
#modeh(1,invented_predicate_cond(var(pos),var(pos)), (anti_reflexive)).

#defined invented_predicate/1.
#defined invented_predicate_cond/2.