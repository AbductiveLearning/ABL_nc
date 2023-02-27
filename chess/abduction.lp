% KB
#include "target.lp".

1{king(X); queen(X); rook(X); bishop(X); knight(X); pawn(X)}1 :- abducible(X).

% at((2,3),_u0).
% at((1,1),_u1).
% sat.
% -sat.

% abducible(_u0).
% queen(_u1).


#show king/1.
#show queen/1.
#show rook/1.
#show bishop/1.
#show knight/1.
#show pawn/1.