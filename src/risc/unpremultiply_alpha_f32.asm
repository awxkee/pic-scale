li {t1}, -1
vsetvli {t1}, {t1}, e32, m1, ta, ma

li   {a7}, 0x3f800000       # Sets 1.0 in {a7} reg

vmv.v.x v5, {a7} # v5 - One in register

fmv.w.x {f1}, x0
vxor.vv v7, v7, v7 # v7 - Set zeros ( single precision )

vlseg4e32.v v1, ({0})

vfdiv.vv v8, v5, v4 # v8 - Reciprocal of alpha

# Multiply RGB by Alpha
vfmul.vv v1, v1, v8
vfmul.vv v2, v2, v8
vfmul.vv v3, v3, v8

vmfeq.vf v0, v4, {f1} # Mask A channel if zero

vmerge.vvm v1, v1, v7, v0
vmerge.vvm v2, v2, v7, v0
vmerge.vvm v3, v3, v7, v0

# Store the result with strided store (interleaving)
vsseg4e32.v v1, ({1})