#max_penalty(1000000).

% KB
succ(X,Y) :- zero(X), one(Y).
succ(X,Y) :- one(X), two(Y).
succ(X,Y) :- two(X), three(Y).
succ(X,Y) :- three(X), four(Y).
succ(X,Y) :- four(X), five(Y).
succ(X,Y) :- five(X), six(Y).
succ(X,Y) :- six(X), seven(Y).
% succ(X,Y) :- seven(X), eight(Y).
% succ(X,Y) :- eight(X), nine(Y).

less(X,Y) :- succ(X,Y).
less(X,Z) :- succ(X,Y), less(Y,Z).
less(X,Y) :- seven(X), nine(Y).

equal(X,Y) :- zero(X), zero(Y).
equal(X,Y) :- one(X), one(Y).
equal(X,Y) :- two(X), two(Y).
equal(X,Y) :- three(X), three(Y).
equal(X,Y) :- four(X), four(Y).
equal(X,Y) :- five(X), five(Y).
equal(X,Y) :- six(X), six(Y).
equal(X,Y) :- seven(X), seven(Y).
% equal(X,Y) :- eight(X), eight(Y).
equal(X,Y) :- nine(X), nine(Y).

-less(X,Y) :- equal(X,Y).
-less(X,Y) :- less(Y,X).

zero(0).
one(1).
two(2).
three(3).
four(4).
five(5).
six(6).
seven(7).
nine(9).






#pos(e0@30,{
less(_u0, _u1)
},{},{
inv_0(_u0).
nine(_u1).
}). 

#pos(e1@30,{
less(_u0, _u1)
},{},{
seven(_u0).
inv_0(_u1).
}). 

#pos(e2@30,{
less(_u0, _u1)
},{},{
inv_0(_u0).
three(_u0).
}). 

#pos(e3@30,{
less(_u0, _u1)
},{},{
inv_0(_u0).
nine(_u1).
}). 

#pos(e4@30,{
less(_u0, _u1)
},{},{
seven(_u0).
inv_0(_u1).
}). 

#pos(e5@30,{
},{less(_u0, _u1)},{
seven(_u0).
inv_0(_u1).
}). 

#pos(e6@30,{

},{less(_u0, _u1)},{
inv_0(_u0).
seven(_u1).
}). 




#modeb(1,less(var(box),var(box)), (anti_reflexive, positive)).
#modeb(1,one(var(digit))).
#modeb(1,two(var(digit))).
#modeb(1,three(var(digit))).
#modeb(1,four(var(digit))).
#modeb(1,five(var(digit))).
#modeb(1,six(var(digit))).
#modeb(1,seven(var(digit))).
#modeb(1,nine(var(digit))).
#modeb(1,inv_0(var(digit))).
#modeh(1,succ(var(digit),var(digit)), (anti_reflexive)).

% Target
% 3 ~ succ(X,Y) :- seven(X), inv_0(Y).
% 3 ~ succ(X,Y) :- inv_0(X), nine(Y).