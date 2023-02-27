div_2(X) :- zero(X).
div_2(X) :- two(X).
div_2(X) :- four(X).
div_2(X) :- six(X).
div_2(X) :- eight(X).
not_div_2(X) :- one(X).
not_div_2(X) :- three(X).
not_div_2(X) :- five(X).
not_div_2(X) :- seven(X).
not_div_2(X) :- nine(X).

div_3(X) :- three(X).
div_3(X) :- six(X).
div_3(X) :- nine(X).
not_div_3(X) :- zero(X).
not_div_3(X) :- one(X).
not_div_3(X) :- two(X).
not_div_3(X) :- four(X).
not_div_3(X) :- five(X).
not_div_3(X) :- seven(X).
not_div_3(X) :- eight(X).

div_4(X) :- four(X).
div_4(X) :- eight(X).
not_div_4(X) :- zero(X).
not_div_4(X) :- one(X).
not_div_4(X) :- two(X).
not_div_4(X) :- three(X).
not_div_4(X) :- five(X).
not_div_4(X) :- six(X).
not_div_4(X) :- seven(X).
not_div_4(X) :- nine(X).

div_5(X) :- five(X).
not_div_5(X) :- zero(X).
not_div_5(X) :- one(X).
not_div_5(X) :- two(X).
not_div_5(X) :- three(X).
not_div_5(X) :- four(X).
not_div_5(X) :- six(X).
not_div_5(X) :- seven(X).
not_div_5(X) :- eight(X).
not_div_5(X) :- nine(X).

% Original definition
even_1(X,Y) :- div_2(X), div_2(Y).
odd_1(X,Y) :- not_div_2(X), not_div_2(Y).
% % Refined definition
% even_1(X,Y) :- div_2(X), div_2(Y), not_div_3(X).
% even_1(X,Y) :- div_2(X), div_2(Y), not_div_3(Y).
% odd_1(X,Y) :- not_div_2(X), not_div_2(Y), not_div_3(X).
% odd_1(X,Y) :- not_div_2(X), not_div_2(Y), not_div_3(Y).
% CWA
misc(X,Y) :- not_div_2(X), div_2(Y).
misc(X,Y) :- div_2(X), not_div_2(Y).
-odd_1(X,Y) :- misc(X,Y).
-even_1(X,Y) :- misc(X,Y).
% Only one class hold
-odd_1(X,Y) :- even_1(X, Y).
-even_1(X,Y) :- odd_1(X, Y).


% Definition
new_1(X,Y) :- div_3(X), div_3(Y).
% Only one class hold
-even_1(X,Y) :- new_1(X,Y).
-new_1(X,Y) :- even_1(X,Y).
-odd_1(X,Y) :- new_1(X,Y).
-new_1(X,Y) :- odd_1(X,Y).
% CWA
-new_1(X,Y) :- misc(X,Y), not_div_3(X).
-new_1(X,Y) :- misc(X,Y), not_div_3(Y).


% For ILASP
% -div_2(X) :- not div_2(X). % CWA (if ==unknown, then ==false)

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
% new_1(_u0,_u1). three(_u0). six(_u1).
