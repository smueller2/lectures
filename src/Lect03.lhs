% CS 340: Programming Paradigms and Patterns
% Lect 03 - Functions
% Michael Lee

> module Lect03 where

Functions
=========

Agenda:
  - Defining functions
    - Pattern matching
    - Guards
    - `where` clause
  - Some useful language constructs
    - `if-else` expressions
    - `case` expressions
    - `let-in` expressions


Defining Functions
------------------

A named, top-level function definition starts with its name, is followed by its
parameter list (separated by spaces), then `=`, then an expression which will 
be evaluated to determine its result.

By convention, we always include type declarations for functions.

> nand :: Bool -> Bool -> Bool
> nand a b = not (a && b)
>
>
> discriminant :: Num a => a -> a -> a -> a
> discriminant a b c = b^2 - 4*a*c
>
>
> c2f :: Fractional a => a -> a
> c2f c = c * 9/5 + 32
 

-- Pattern matching

We can provide multiple alternative expressions to be evaluated when a function
is called, which are differentiated based on *patterns* that are matched against
the parameter values. Patterns are matched in order (top to bottom); only the
first pattern to match has its corresponding expression evaluated.

> not' :: Bool -> Bool
> not' True = False
> not' False = True


A catch-all pattern, where a variable is specified instead of a data value, can 
be used to match parameters not specified in earlier patterns.

> fib :: Integer -> Integer
> fib 0 = 1
> fib 1 = 1
> fib n = fib (n-1) + fib (n-2)


We can also use the wildcard pattern `_` to match on one or more values we don't
care about.

> nand' :: Bool -> Bool -> Bool
> nand' True True = False
> nand' _    _    = True


Patterns can also be used to "deconstruct" values. E.g., for tuples:

> fst' :: (a,b) -> a
> fst' (x,_) = x
>
> 
> distance :: (Floating a, Eq a) => (a,a) -> (a,a) -> a
> distance (x1,y1) (x2,y2) = sqrt ((x1-x2)^2 + (y1-y2)^2)
>
>
> mapTup :: (a -> b) -> (a,a) -> (b,b)
> mapTup f (x,y) = (f x, f y)


-- Guards

Boolean expressions can be used to provide more granular *guards* for separate
equations in a function definition. The `otherwise` keyword (which is just 
`True` in disguise) can be used to provide a catch-all equation.

> fib' :: Integer -> Integer
> fib' n | n == 0 = 0
>        | n == 1 = 1
>        | otherwise = fib' (n-1) + fib' (n-2)
>
>
> c2h :: (Floating a, Ord a) => a -> String
> c2h c | c2f c < 0   = "too cold"
>       | c2f c > 100 = "too hot"
>       | otherwise   = "tolerable"
>
>
> quadRoots :: (Floating a, Ord a) => a -> a -> a -> (a, a)
> quadRoots a b c 
>   | discriminant a b c >= 0 = ((-b + sqrt (discriminant a b c)) / (2*a),
>                                (-b - sqrt (discriminant a b c)) / (2*a))
>   | otherwise = error "No real roots"


-- `where` clause

A `where` clause lets us introduce new local bindings (vars or functions) in a 
given function definition (which may span multiple guards, but *not* separate 
top-level patterns). Note that we read the `|` symbol as "such that".

> quadRoots' :: (Floating a, Ord a) => a -> a -> a -> (a, a)
> quadRoots' a b c 
>     | d >= 0 = ((-b + sqrt_d) / (2*a), (-b - sqrt_d) / (2*a))
>     | otherwise = error "No real roots"
>   where disc a b c = b^2 - 4*a*c
>         d          = disc a b c
>         sqrt_d     = sqrt d


Some useful language constructs
-------------------------------

An important note about the following constructs: they are all used to create
*expressions* --- i.e., they evaluate to values (which must have a consistent,
static type regardless of the evaluation path). They are not statements!


-- `if-else` expressions

The classic conditional (but both paths must exist, and must also evaluate to 
the same type!)

> -- try:
> -- oneOrOther n = if n < 0 then True else "False"
>
>
> fib'' :: Integer -> Integer
> fib'' n = if n <= 1 
>           then 1 
>           else fib'' (n-1) + fib'' (n-2)


-- `case` expressions

`case` expressions allow us to perform pattern matching --- just as we can 
across top-level function definitions --- on an arbitrary expression. Patterns
can also be followed by guards!

> greet :: String -> String
> greet name = "Hello" ++ 
>              case name of "Michael" -> " and welcome!"
>                           "Tom"     -> " friend."
>                           "Harry"   -> " vague acquaintance."
>                           name | null name -> " nobody."
>                                | otherwise -> " stranger."
>              ++ " How are you?"


-- `let-in` expressions

Similar to a `where` clause, the `let` keyword allows us to create new 
bindings, but only usable within the expression following the `in` keyword.
The entire `let-in` construct is also an *expression* --- it evaluates to the
value of the expression following `in`.

> quadRoots'' :: (Floating a, Ord a) => a -> a -> a -> (a, a)
> quadRoots'' a b c = let disc a b c = b^2 + 4*a*c
>                         d          = disc a b c
>                         sqrt_d     = sqrt d
>                     in if d >= 0 
>                        then ((-b + sqrt_d) / (2*a), (-b - sqrt_d) / (2*a))
>                        else error "No real roots"
>
>
> dist2h :: (Floating a, Ord a, Show a) => (a,a) -> String
> dist2h p = "The distance is " ++
>            let (x,y) = p
>                d = sqrt (x^2 + y^2)
>            in if d > 100 then "too far" else show d
