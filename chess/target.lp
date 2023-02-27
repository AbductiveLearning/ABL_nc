
succ(X,Y) :- zero(X), one(Y).
succ(X,Y) :- one(X), two(Y).
succ(X,Y) :- two(X), three(Y).
succ(X,Y) :- three(X), four(Y).
succ(X,Y) :- four(X), five(Y).
succ(X,Y) :- five(X), six(Y).
succ(X,Y) :- six(X), seven(Y).

equal(X,Y) :- zero(X), zero(Y).
equal(X,Y) :- one(X), one(Y).
equal(X,Y) :- two(X), two(Y).
equal(X,Y) :- three(X), three(Y).
equal(X,Y) :- four(X), four(Y).
equal(X,Y) :- five(X), five(Y).
equal(X,Y) :- six(X), six(Y).
equal(X,Y) :- seven(X), seven(Y).

less(X,Y) :- succ(X,Y).
less(X,Z) :- less(X,Y), less(Y,Z).

% num(0).
% num(1).
% num(2).
% num(3).
% num(4).
% num(5).
% num(6).
% num(7).
% num(8).
% num(9).
% succ(X,Y) :- X+1 = Y, num(X), num(Y).
% equal(X,Y) :- X = Y, num(X), num(Y).
% less(X,Y) :- X < Y, num(X), num(Y).

ic :- at((X1,Y1),Type1), at((X2,Y2),Type2), attack((X1,Y1),Type1,(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- king(Type1), one_step((X1,Y1),(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- queen(Type1), straight_or_diag((X1,Y1),(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- rook(Type1), straight((X1,Y1),(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- bishop(Type1), diag((X1,Y1),(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- knight(Type1), lshape((X1,Y1),(X2,Y2)).
attack((X1,Y1),Type1,(X2,Y2)) :- pawn(Type1), diag_forward((X1,Y1),(X2,Y2)).

left((X1,Y1),(X2,Y2)) :- succ(X2,X1), equal(Y1,Y2).
right((X1,Y1),(X2,Y2)) :- succ(X1,X2), equal(Y1,Y2).
forward((X1,Y1),(X2,Y2)) :- succ(Y1,Y2), equal(X1,X2).
backward((X1,Y1),(X2,Y2)) :- succ(Y2,Y1), equal(X1,X2).
left_forward((X1,Y1),(X2,Y2)) :- left((X1,Y1),(X3,Y3)), forward((X3,Y3),(X2,Y2)).
right_forward((X1,Y1),(X2,Y2)) :- right((X1,Y1),(X3,Y3)), forward((X3,Y3),(X2,Y2)).
left_backward((X1,Y1),(X2,Y2)) :- left((X1,Y1),(X3,Y3)), backward((X3,Y3),(X2,Y2)).
right_backward((X1,Y1),(X2,Y2)) :- right((X1,Y1),(X3,Y3)), backward((X3,Y3),(X2,Y2)).

one_step((X1,Y1),(X2,Y2)) :- left((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- right((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- forward((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- backward((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- left_backward((X1,Y1),(X2,Y2)).
one_step((X1,Y1),(X2,Y2)) :- right_backward((X1,Y1),(X2,Y2)).
straight_or_diag((X1,Y1),(X2,Y2)) :- straight((X1,Y1),(X2,Y2)).
straight_or_diag((X1,Y1),(X2,Y2)) :- diag((X1,Y1),(X2,Y2)).
straight((X1,Y1),(X2,Y2)) :- equal(X1,X2), less(Y1,Y2).
straight((X1,Y1),(X2,Y2)) :- equal(X1,X2), less(Y2,Y1).
straight((X1,Y1),(X2,Y2)) :- equal(Y1,Y2), less(X1,X2).
straight((X1,Y1),(X2,Y2)) :- equal(Y1,Y2), less(X2,X1).
diag((X1,Y1),(X2,Y2)) :- diag_0((X1,Y1),(X2,Y2)).
diag((X1,Y1),(X2,Y2)) :- diag_1((X1,Y1),(X2,Y2)).
diag((X1,Y1),(X2,Y2)) :- diag_2((X1,Y1),(X2,Y2)).
diag((X1,Y1),(X2,Y2)) :- diag_3((X1,Y1),(X2,Y2)).
diag_0((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X2,Y2)).
diag_0((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X3,Y3)), diag_0((X3,Y3),(X2,Y2)).
diag_1((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X2,Y2)).
diag_1((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X3,Y3)), diag_1((X3,Y3),(X2,Y2)).
diag_2((X1,Y1),(X2,Y2)) :- left_backward((X1,Y1),(X2,Y2)).
diag_2((X1,Y1),(X2,Y2)) :- left_backward((X1,Y1),(X3,Y3)), diag_2((X3,Y3),(X2,Y2)).
diag_3((X1,Y1),(X2,Y2)) :- right_backward((X1,Y1),(X2,Y2)).
diag_3((X1,Y1),(X2,Y2)) :- right_backward((X1,Y1),(X3,Y3)), diag_3((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X3,Y3)), left((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X3,Y3)), forward((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X3,Y3)), right((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X3,Y3)), forward((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- left_backward((X1,Y1),(X3,Y3)), left((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- left_backward((X1,Y1),(X3,Y3)), backward((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- right_backward((X1,Y1),(X3,Y3)), right((X3,Y3),(X2,Y2)).
lshape((X1,Y1),(X2,Y2)) :- right_backward((X1,Y1),(X3,Y3)), backward((X3,Y3),(X2,Y2)).
diag_forward((X1,Y1),(X2,Y2)) :- left_forward((X1,Y1),(X2,Y2)).
diag_forward((X1,Y1),(X2,Y2)) :- right_forward((X1,Y1),(X2,Y2)).


zero(0).
one(1).
two(2).
three(3).
four(4).
five(5).
six(6).
seven(7).

king(ki).
queen(q).
rook(r).
bishop(b).
knight(kn).
pawn(p).

sat :- not ic. % Default is sat (if ic==unknown, then sat==true)
-sat :- ic. % if ic==true, then sat==false
% :- sat, ic. % Cannot be true at the same time
% -sat :- not sat. % CWA (if sat==unknown, then sat==false)

% Test

% % at((1,1),q).
% % at((4,2),kn).
% at((1,1),kn).
% at((1,2),ki).
% sat. 

% at((1,1),_u0). at((1,2),_u1). 
% knight(_u0). king(_u1).
% -sat.
