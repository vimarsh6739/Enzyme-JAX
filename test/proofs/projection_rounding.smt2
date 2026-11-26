; Two steps, seed 1, decay float32(0.9), feature float32(1.2).
; The negation of the reported pair of IEEE binary32 results is unsatisfiable.
; RNE after every operation; a backend using contraction may round differently.
(set-logic QF_FP)
(define-fun x () Float32 ((_ to_fp 8 24) #x3f99999a))
(define-fun a () Float32 ((_ to_fp 8 24) #x3f666666))
(define-fun one () Float32 ((_ to_fp 8 24) #x3f800000))
(define-fun old () Float32 (fp.add RNE (fp.mul RNE one x) (fp.mul RNE a x)))
(define-fun new () Float32 (fp.mul RNE (fp.add RNE one a) x))
(assert (not (and (= old ((_ to_fp 8 24) #x4011eb86))
                  (= new ((_ to_fp 8 24) #x4011eb85)))))
(check-sat)
