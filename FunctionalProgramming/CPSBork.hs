{-|
Module: CPSBork Assignment 2
Description: Continuation Passing Style Transformations
Copyright: (c) University of Toronto, 2019
               CSC324 Principles of Programming Languages, Fall 2019

-}
-- This lists what this module exports. Don't change this!

module CPSBork (
    -- Warmup
    cpsFacEnv, fibEnv, cpsFibEnv,
    -- CPS Transform
    cpsDef, cpsExpr
) where

import qualified Data.Map as Map
import Test.QuickCheck (quickCheck)
import Ex10Bork (Env, emptyEnv, Value(..), HaskellProc(..), Expr(..), eval, def)

------------------------------------------------------------------------------
-- Warmup
------------------------------------------------------------------------------

-- | facEnv is an environment containing the function `fac` that computes the
--   factorial of a number, written in direct style.
facEnv :: Env
facEnv = def [("fac", Lambda ["n"]
                (If (Equal (Var "n") (Literal $ Num 0))
                    (Literal $ Num 1)
                    (Times (Var "n") (App (Var "fac")
                       [(Plus (Var "n") (Literal $ Num (-1)))]))))]

-- | cpsFacEnv is an environment containing the function `cps_fac` that computes the
--   factorial of a number, written in CPS
cpsFacEnv :: Env
cpsFacEnv = def [("cps_fac", Lambda ["n", "k"]
              (If (Equal (Var "n") (Literal $ Num 0)) -- condition does not change
                    (App (Var "k") [Literal $ Num 1]) -- base case
                    (App (Var "cps_fac") [
                    (Plus (Var "n") (Literal $ Num (-1))),  -- apply cpsFac to n-1 and k
                    (Lambda ["r"] (App (Var "k") [(Times (Var "n") (Var "r"))]))
                    ])))]



-- | fibEnv is an environment containing the function `fib` that computes the
--   n-th fibonacci via recursion, written in direct style.
fibEnv :: Env
fibEnv = def [("fib", Lambda ["n"]
                (If (Equal (Var "n") (Literal $ Num 0))
                    (Literal $ Num 0)
                    (If (Equal (Var "n") (Literal $ Num 1))
                    (Literal $ Num 1)
                    (Plus
                    (App (Var "fib") [(Plus (Var "n") (Literal $ Num (-1)))])
                    (App (Var "fib") [(Plus (Var "n") (Literal $ Num (-2)))])
                    )
                    )))]

-- | cpsFfibEnv is an environment containing the function `cps_fib` that computes the
--   n-th fibonacci via recursion, written in CPS
cpsFibEnv :: Env
cpsFibEnv = def [("cps_fib", Lambda ["n", "k"]
                (If (Equal (Var "n") (Literal $ Num 0))
                    (App (Var "k") [Literal $ Num 0])
                    (If (Equal (Var "n") (Literal $ Num 1))
                    (App (Var "k") [Literal $ Num 1])
                    (App (Var "cps_fib") [
                        (Plus (Var "n") (Literal $ Num (-1))),
                        (Lambda ["x"] (App (Var "cps_fib")[
                              (Plus (Var "n") (Literal $ Num (-2))),
                              (Lambda ["y"] (App (Var "k")[
                              (Plus (Var "x") (Var "y"))]
                              ))]))]))))]

-- | An identity function in Bork, used for testing
identityFn :: Expr
identityFn = Lambda ["x"] (Var "x")

-- | Some simple tests. You should write your own.

prop_testFac :: Bool
prop_testFac = eval facEnv (App (Var "fac") [Literal $ Num 3]) == Num 6
prop_testCpsFac :: Bool
prop_testCpsFac = eval cpsFacEnv (App (Var "cps_fac") [Literal $ Num 3, identityFn]) == Num 6
prop_testFib :: Bool
prop_testFib = eval fibEnv (App (Var "fib") [Literal $ Num 6]) == Num 8
prop_testCpsFib :: Bool
prop_testCpsFib = eval cpsFibEnv (App (Var "cps_fib") [Literal $ Num 6, identityFn]) == Num 8
------------------------------------------------------------------------------
-- CPS Transformation
------------------------------------------------------------------------------

-- | Performs CPS Transformations on a list of name -> expression bindings
-- by renaming the names, and CPS transforming the expressions
cpsDef :: [(String, Expr)] -> [(String, Expr)]
cpsDef bindings = map (\(s, e) -> (rename s, cpsExpr e "" id)) bindings

-- | CPS Transform a single expression
cpsExpr :: Expr -> String -> (Expr -> Expr) -> Expr
-- literals:
cpsExpr (Literal v) s context = context $ Literal $ v
-- variables:
cpsExpr (Var name)  s context = context $ Var $ rename name
-- builtins:s
cpsExpr (Plus left right)  s context =
  cpsExpr left (s ++ "PlusLeft") (\l -> cpsExpr right (s ++ "PlusRight") (\r -> context $ Plus l r))

cpsExpr (Times left right) s context =
  cpsExpr left (s ++ "TimesLeft") (\l -> cpsExpr right (s ++ "TimesRight") (\r -> context $ Times l r))

cpsExpr (Equal left right) s context =
  cpsExpr left (s ++ "EqualLeft") (\l -> cpsExpr right (s ++ "EqualRight") (\r -> context $ Equal l r))

-- function definition:
cpsExpr (Lambda params body) s context =
  context (Lambda (renameVars params) (cpsExpr body (s ++ "Lambda") (\cB -> App (Var "k") [cB])))
-- function application:
cpsExpr (App fn args) s context =
  cpsExpr fn s (\cF ->
  cpsArgs args s [] (\cArgs ->
    App cF (cArgs ++  [(Lambda ["Result" ++ s] (context (Var ("Result" ++ s))))])
  ))
-- if expressions
cpsExpr (If cond conseq altern) s context =
  cpsExpr cond (s ++ "Cond") (\cCond ->
  (If cCond
    (cpsExpr conseq (s ++ "conseq") context)
    (cpsExpr altern (s ++ "Altern") context)
    )
  )

-- | Helper function that renames a variable by prepending "cps_"
rename :: String -> String
rename s = "cps_" ++ s

-- | Helper function that converts lambda params to cps, and adds "k"
renameVars :: [String] -> [String]
renameVars [] = ["k"]
renameVars (x:xs) = [rename x] ++ (renameVars xs)

--cpsArgs :: [Expr] -> String -> [Expr] -> (Expr -> Expr)
cpsArgs [] s vargs k = k vargs
cpsArgs (x:xs) s vargs k =
  cpsExpr x ("Result" ++  s) (\varg ->
  cpsArgs xs s (vargs ++ [varg]) k)


-- | Some simple tests.

prop_testCpsExprLiteral :: Bool
prop_testCpsExprLiteral = result == Num 1
    where bindings = cpsDef [("n", Literal $ Num 1)]
          env = def bindings
          result = eval env $ Var ("cps_n")

prop_testCpsExprVar :: Bool
prop_testCpsExprVar = result == Num 2
    where bindings = cpsDef [("n", Literal $ Num 2),
                             ("m", Var "n")]
          env = def bindings
          result = eval env $ Var ("cps_m")

prop_testCpsExprPlus :: Bool
prop_testCpsExprPlus = result == Num 5
    where bindings = cpsDef [("n", Literal $ Num 2),
                             ("m", (Plus (Var "n") (Literal $ Num 3)))]
          env = def bindings
          result = eval env $ Var "cps_m"

prop_testCpsExprTimes :: Bool
prop_testCpsExprTimes = result == Num 6
    where bindings = cpsDef [("n", Literal $ Num 2),
                             ("m", (Times (Var "n") (Plus (Var "n") (Literal $ Num 1))))]
          env = def bindings
          result = eval env $ Var "cps_m"

prop_testCpsExprEqualTrue :: Bool
prop_testCpsExprEqualTrue = result == T
    where bindings = cpsDef [("n", Literal $ Num 3),
                             ("m", (Equal (Var "n") (Literal $ Num 3)))]
          env = def bindings
          result = eval env $ Var "cps_m"

prop_testCpsTimesAppLambda :: Bool
prop_testCpsTimesAppLambda = result == Num 6
    where bindings = cpsDef [("n", Literal $ Num 2),
                             ("m", (Times (Literal $ Num 2) (App (Lambda ["n"]
                             (Var "n")) [Literal $ Num 3])))]
          env = def bindings
          result = eval env $ Var "cps_m"

prop_testCpsExprEqualFalse :: Bool
prop_testCpsExprEqualFalse = result == F
    where bindings = cpsDef [("n", Literal $ Num 2),
                             ("m", (Equal (Var "n") (Literal $ Num 3)))]
          env = def bindings
          result = eval env $ Var "cps_m"

prop_testCpsExprFac :: Bool
prop_testCpsExprFac = result == Num 720
    where bindings = cpsDef [("fac", Lambda ["n"]
                                (If (Equal (Var "n") (Literal $ Num 0))
                                    (Literal $ Num 1)
                                    (Times (Var "n") (App (Var "fac")
                                       [(Plus (Var "n") (Literal $ Num (-1)))]))))]
          env = def bindings
          result = eval env $ (App (Var "cps_fac") [Literal $ Num 6, identityFn])

prop_testCpsExprFib :: Bool
prop_testCpsExprFib = result == Num 144
    where bindings = cpsDef [("fib", Lambda ["n"]
                    (If (Equal (Var "n") (Literal $ Num 0))
                        (Literal $ Num 0)
                        (If (Equal (Var "n") (Literal $ Num 1))
                        (Literal $ Num 1)
                        (Plus
                        (App (Var "fib") [(Plus (Var "n") (Literal $ Num (-1)))])
                        (App (Var "fib") [(Plus (Var "n") (Literal $ Num (-2)))])
                        )
                        )))]
          env = def bindings
          result = eval env $ (App (Var "cps_fib") [Literal $ Num 12, identityFn])

prop_testLambdaSimple :: Bool
prop_testLambdaSimple =
  (cpsExpr (Lambda ["x", "y"] (Plus (Var "x") (Var "y"))) "" id) ==
    (Lambda ["cps_x","cps_y","k"] (App (Var "k") [Plus (Var "cps_x") (Var "cps_y")]))

prop_testAppSimple :: Bool
prop_testAppSimple =
  (cpsExpr (Plus (Literal $ Num 1) (App (Var "f") [Plus (Var "x") (Var "y"), Var "y"])) "" id) ==
    (App (Var "cps_f") [Plus (Var "cps_x") (Var "cps_y"),
      Var "cps_y",
      Lambda ["ResultPlusRight"] (Plus (Literal (Num 1)) (Var "ResultPlusRight"))])


------------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------------

-- | This main function runs the quickcheck tests.
-- This gets executed when you compile and run this program. We'll talk about
-- "do" notation much later in the course, but for now if you want to add your
-- own tests, just define them above, and add a new `quickcheck` line here.
main :: IO ()
main = do
    quickCheck prop_testFac
    quickCheck prop_testCpsFac
    quickCheck prop_testFib
    quickCheck prop_testCpsFib
    quickCheck prop_testCpsExprLiteral
    quickCheck prop_testCpsExprVar
    quickCheck prop_testCpsExprPlus
    quickCheck prop_testCpsExprTimes
    quickCheck prop_testCpsExprEqualTrue
    quickCheck prop_testCpsExprEqualFalse
    quickCheck prop_testCpsTimesAppLambda
    quickCheck prop_testCpsExprFac
    quickCheck prop_testCpsExprFib
    quickCheck prop_testLambdaSimple
    quickCheck prop_testAppSimple
