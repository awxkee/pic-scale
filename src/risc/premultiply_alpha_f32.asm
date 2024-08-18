li {t1}, -1
vsetvli {t1}, {t1}, e32, m1, ta, ma

vlseg4e32.v v0, ({0})

# Multiply RGB by Alpha
vfmul.vv v0, v0, v3
vfmul.vv v1, v1, v3
vfmul.vv v2, v2, v3

# Store the result with strided store (interleaving)
vsseg4e32.v v0, ({1})