#lang racket #| * CSC324 Fall 2019: Assignment 1 * |#
#|
Module: dubdub
Description: Assignment 1: A More Featureful Interpreter
Copyright: (c)University of Toronto, University of Toronto Mississauga 
               CSC324 Principles of Programming Languages, Fall 2019

The assignment handout can be found at

    https://www.cs.toronto.edu/~lczhang/324/files/a1.pdf

Please see the assignment guidelines at 

    https://www.cs.toronto.edu/~lczhang/324/homework.html
|#

(provide run-interpreter)

(require "dubdub_errors.rkt")


;-----------------------------------------------------------------------------------------
; Main functions (skeleton provided in starter code)
;-----------------------------------------------------------------------------------------
#|
(run-interpreter prog) -> any
  prog: datum?
    A syntactically-valid Dubdub program.

  Evaluates the Dubdub program and returns its value, or raises an error if the program is
  not semantically valid.
|#
(define (run-interpreter prog) 
  (if (number? (first prog)) (first prog) ;Base case, return number if prog is just a number
      (if (boolean? (first prog)) (first prog) ;Base case, return boolean if prog is just a boolean
          (let* ([first-prog (first prog)])
            (cond
              [(or (equal? (first first-prog) 'define) (equal? (first first-prog) 'define-contract)) (define-interpreter prog (hash))] ;Run helper on
              ; definitions and contracts
              [(empty? (rest prog)) (interpret (hash) first-prog)]
              [else (run-interpreter (rest prog))])
            )
          )
      )
  )

#|
(interpret env expr) -> any
  env: hash?
    The environment with which to evaluate the expression.
  expr: datum?
    A syntactically-valid Dubdub expression.

  Returns the value of the Dubdub expression under the given environment.
|#
(define (interpret env expr)
  (cond
    [(number? expr) expr] ;Base case, returns expr if expr is just a number
    [(boolean? expr) expr] ;Base case, returns expr if expr is just a boolean
    [(and (symbol? expr) (not (equal? expr 'closure))) (hash-ref-helper env expr)] ;Base case, returns value if expr is a key in env
    [else
     (let ([first-element (first expr)])
       (cond
         [(procedure? env first-element) ;evaluates function calls
          (let* (
                 [function-call first-element]
                 [closure (hash-ref env function-call)]
                 [arguments (rest expr)]
                 [body (second closure)]
                 [function (third body)]
                 [parameters (param-checker (second body))]
                 [env-closure (third closure)]
                 [new-env (lambda-bindings parameters arguments env-closure)]
                 )       
            (cond
              [(< (length arguments) (length parameters)) ;Handles the case in which functions should be curried
               (let* (
                      [new-params (curry-params parameters new-env)]
                      [n-function (curry-function (rest function) new-env (list))]
                      [new-function (append (cons (first function) (list)) n-function)]
                      [new-body (cons 'lambda (cons new-params (cons new-function (list))))]
                      [new-closure (cons 'closure (cons new-body (cons new-env (list))))])
                 new-closure)]
              [(equal? (length arguments) (length parameters)) ;Evaluate functions when given correct number of arguments
               (cond
                 [(equal? 4 (length closure)) ;Handles the case in which the closure has a contract
                  (let* (
                         [contract (first (rest (fourth closure)))]
                         [pre (first contract)]
                         [post (second contract)]
                         [ret-val (interpret new-env function)])
                    (cond
                      [(and (pre-helper pre arguments new-env) (post-helper post ret-val new-env)) ret-val]
                      [else (report-error 'contract-violation)] ;Reports error in the case that at least on of the conditions is not met
                    ))
                  ]
                 [else (interpret new-env function)])
               ]
              [else (report-error 'arity-mismatch (length arguments) (length parameters))] ;Handles the case in which their are more arguments than required         
              )
            )]
         [(builtin? first-element) ;runs builtins functions
          (cond
            [(equal? first-element '+) (add-helper env (rest expr) 0)] ;Runs helper function to handle addition of more than two elements
            [(or (equal? first-element 'equal?) (equal? first-element '<)) ;checks to see if builtin takes two arguments
             (let* ([arg1 (second expr)]
                    [arg2 (third expr)])
               ((hash-ref builtins first-element) (interpret env arg1) (interpret env arg2)))]
            [else ((hash-ref builtins first-element) (interpret env (second expr)))])] ;runs on builtins which only takes one argument         
         [(equal? first-element 'closure) ;evaluates closures
          (let* (
                 [closure expr]
                 [body (second closure)]
                 [function (third body)]
                 [new-env (third closure)])
            (interpret new-env function))
          ]
         [(equal? first-element 'lambda) ;creates closure for function definition
          (let* (
                 [body expr]
                 )
            (cons 'closure (cons body (cons env (list)))))
          ]
         [(equal? (first first-element) 'lambda) ;create closure from lambda unary function with format '('closure (lambda body) (environment))
          (let* ([body first-element]
                 [keys (second first-element)]
                 [values (rest expr)]
                 [new-env (lambda-bindings keys values env)]
                 [closure (cons 'closure (cons body (cons new-env (list))))])            
            (interpret new-env closure))
          ]
         [else (report-error 'not-a-function (first (first-element)))]
         )
       )
     ]
    )
  )


;-----------------------------------------------------------------------------------------
; Helpers: Builtins and closures
;-----------------------------------------------------------------------------------------

; Function to make a contract. returns a list which contains a list of preconditions and a postcondition: ((<pre> ...) <post>)
(define (make-contract contract pre post)
  (cond
    [(not (empty? post)) (cons pre (cons post (list)))]
    [else
     (cond
       [(equal? (first contract) '->) (make-contract (rest contract) pre (append post (list (second contract))))]
       [else
        (make-contract (rest contract) (append pre (list (first contract))) post)
        ]
       )
     ]
    ))

; Checks if the paramaters declared in a function definiton do not contain any duplicates. Raises duplicate-name error if it does.
(define (param-checker params)
  (cond
    [(not (check-duplicates params)) params]
    [else (report-error 'duplicate-name (check-duplicates params))]
    ))
  

;Checks that pre-conditions in contract hold, returns True if they do, False if they don't
(define (pre-helper pre arguments env)
  (cond
    [(and (equal? (length arguments) (length pre)) (empty? arguments)) #t]
    [else
     (let* (
            [function (first pre)]
            [arg (first arguments)]
            [f-call (cons function (cons arg (list)))])
       (cond
         [(equal? function 'any) (pre-helper (rest pre) (rest arguments) env)]
         [else
          (let* (
                 [ret-val (interpret env f-call)])
            (cond
              [ret-val (pre-helper (rest pre) (rest arguments) env)]
              [else #f])
            )
          ]
         ))]))

;Checks that post-condition of the contract holds. Returns True if it does, False otherwise
(define (post-helper post ret-val env)
  (let* (
         [function (first post)]
         [f-call (cons function (cons ret-val (list)))])
    (cond
      [(not (equal? (length post) 1)) #f]
      [(equal? function 'any) #t]
      [(interpret env f-call) #t]
      [else #f])
         ))

; Checks if a key in the given environment is bound to a procedure.
(define (procedure? env identifier)
  (if (not (hash-has-key? env identifier)) #f
      (let* ([value (hash-ref env identifier)])
        (cond
          [(list? value) (equal? (first value) 'closure)]
          [else #f])
        )))

; A hash mapping symbols for Dubdub builtin functions to their corresponding Racket value.
(define builtins
  (hash
   '+ +
   'equal? equal?
   '< <
   'integer? integer?
   'boolean? boolean?
   'procedure? procedure?
   ))

; Returns whether a given symbol refers to a builtin Dubdub function.
(define (builtin? identifier) (hash-has-key? builtins identifier))

; Given list of expressions, returns sum of all elements
(define (add-helper env expr acc)
  (cond
    [(number? expr) expr]
    [(not (empty? expr))
     (let* (
            [new-acc (+ acc (interpret env (first expr)))])
       (add-helper env (rest expr) new-acc)
       )
     ]
    [else acc]
    )
  )

;Takes a list of parameters and a hash table, returns a new list with parameters replaced by those in hash table
(define (curry-params params env)
  (cond
    [(hash-has-key? env (first params))
     (curry-params (rest params) env)]
    [else params]                 
    ))

;Takes a function body, and returns a new one in terms of the arguments given in the environment.
(define (curry-function function env new-function)
  (cond
    [(hash-has-key? env (first function))
     (let* (
            [value (hash-ref env (first function))]
            [l-value (cons value (list))]
            [n-function (append new-function l-value)])
       (curry-function (rest function) env n-function)
       )
     ]
    [else (append new-function function)]
    ))


; Binds keys to values, returns updated map. Bindings given in list of key/value pairs: '((a 2) (b 1))
(define (bind-vars bindings map)
  (cond
    [(empty? bindings) map]
    [else
     (let* ([defn (first bindings)]
            [key (first defn)]
            [value (interpret map (second defn))]
            [new-env (hash-set map key value)])
       (bind-vars (rest bindings) new-env)
       )]))

; Binds keys to values for lambda functions, returns updated map. Bindings given in list of keys and a list of values: '((a b) (1 2))
(define (lambda-bindings keys values map)
  (cond
    [(or (empty? keys) (empty? values)) map]
    [else
     (let* ([key (first keys)]
            [value (interpret map (first values))]
            [new-env (hash-set map key value)])
       (lambda-bindings (rest keys) (rest values) new-env)
       )]))

; Performs hash-ref function after checking for unbound-name error.
(define (hash-ref-helper env key)
  (cond
    [(hash-has-key? env key) (interpret env (hash-ref env key))]
    [else (report-error 'unbound-name key)]
    )
  )

; Checks if the given function is bound to a contract
(define (has-contract? env key)
  (cond
    [(not (hash-has-key? env key)) #f]
    [else (equal? (first (hash-ref env key)) 'contract)])
  )

; Handles case in which program contains bindings and contracts.
(define (define-interpreter prog env)
  (cond
    [(empty? prog) env]
    [(symbol? (first prog)) (hash-ref-helper env (first prog))]
    [(equal? (first (first prog)) 'define-contract)
     (let* (
            [definition (first prog)]
            [key (second definition)]
            [contract (third definition)]
            [new-contract (cons 'contract (cons (make-contract contract (list) (list)) (list)))]
            )
       (cond
         [(hash-has-key? env key) (report-error 'invalid-contract key)] ;multiple top level definitions error should be raised.
         [else
          (let* (                 
                 [new-env (hash-set env key new-contract)])
            (define-interpreter (rest prog) new-env)
            )]
         )
       )
     ]
    [(equal? (first (first prog)) 'define) ;Handles the case in which the given program is a binding
     (let* (
            [definition (first prog)]
            [key (second definition)]
            [value (third definition)])
       (cond
         [(has-contract? env key)
          (let* (
                 [contract (hash-ref env key)]
                 [old-env (hash-remove env key)]
                 [closure (cons 'closure (cons value (cons old-env (cons contract (list)))))]
                 [new-env (hash-set old-env key closure)])
            (define-interpreter (rest prog) new-env))
          ]
         [(hash-has-key? env key) (report-error 'duplicate-name key)] ;multiple top level definitions error should be raised.
         [else
          (let* (
                 [binding (cons (cons key (cons value (list)))(list))]
                 [new-env (bind-vars binding env)])
            (define-interpreter (rest prog) new-env)
            )]))]
    [(procedure? env (first (first prog)))
     (interpret env (first prog))]
    [(builtin? (first (first prog))) (interpret env (first prog))]
    [else (report-error 'not-a-function (first (first prog)))]
    )
  )

#|
Starter definition for a closure "struct". Racket structs behave similarly to
C structs (contain fields but no methods or encapsulation).
Read more at https://docs.racket-lang.org/guide/define-struct.html.

You can and should modify this as necessary. If you're having trouble working with
Racket structs, feel free to switch this implementation to use a list/hash instead.
|#
(struct closure (params body))
