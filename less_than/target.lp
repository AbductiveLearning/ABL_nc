
succ(X,Y) :- zero(X), one(Y).
succ(X,Y) :- one(X), two(Y).
succ(X,Y) :- two(X), three(Y).
succ(X,Y) :- three(X), four(Y).
succ(X,Y) :- four(X), five(Y).
succ(X,Y) :- five(X), six(Y).
succ(X,Y) :- six(X), seven(Y).
succ(X,Y) :- seven(X), eight(Y).
succ(X,Y) :- eight(X), nine(Y).

less(X,Y) :- succ(X,Y).
less(X,Z) :- less(X,Y), less(Y,Z).

% For ILASP
less(X,Y) :- zero(X), two(Y).
less(X,Y) :- one(X), three(Y).
less(X,Y) :- two(X), four(Y).
less(X,Y) :- three(X), five(Y).
less(X,Y) :- four(X), six(Y).
less(X,Y) :- five(X), seven(Y).
less(X,Y) :- six(X), eight(Y).
less(X,Y) :- seven(X), nine(Y).

equal(X,Y) :- zero(X), zero(Y).
equal(X,Y) :- one(X), one(Y).
equal(X,Y) :- two(X), two(Y).
equal(X,Y) :- three(X), three(Y).
equal(X,Y) :- four(X), four(Y).
equal(X,Y) :- five(X), five(Y).
equal(X,Y) :- six(X), six(Y).
equal(X,Y) :- seven(X), seven(Y).
equal(X,Y) :- eight(X), eight(Y).
equal(X,Y) :- nine(X), nine(Y).

-less(X,Y) :- equal(X,Y).
-less(X,Y) :- less(Y,X).

% -less(X,Y) :- not less(X,Y). % CWA (if less(X,Y)==unknown, then less(X,Y)==false)

zero(0).
one(1).
two(2).
three(3).
four(4).
five(5).
six(6).
seven(7).
eight(8).
nine(9).


% Test
% less(_u0,_u1). four(_u0). three(_u1).
