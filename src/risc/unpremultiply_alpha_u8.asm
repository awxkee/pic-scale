vsetvli zero, {2}, e8, m1, ta, ma

li   t4, 255

vlseg4e8.v v0, ({0})

vmv.v.v v10, v0

vmseq.vi v0, v3, 0       # Erase to zero mask
li t5, 1
vmerge.vxm v3, v3, t5, v0
vmv.v.v v0, v10

# Widening lower parts
vwmulu.vx v4, v0, t4     # R16
vwmulu.vx v6, v1, t4     # G16
vwmulu.vx v8, v2, t4     # B16
vwaddu.vx v10, v3, x0    # A16

srli t5, {2}, 1

vsetvli zero, t5, e16, m1, ta, ma

# Div by alpha

vdivu.vv v4, v4, v10
vdivu.vv v5, v5, v11
vdivu.vv v6, v6, v10
vdivu.vv v7, v7, v11
vdivu.vv v8, v8, v10
vdivu.vv v9, v9, v11

vsetvli zero, {2}, e8, m1, ta, ma

# Narrowing

vnclipu.wi v10, v4, 0   # R8
vnclipu.wi v1, v6, 0    # G8
vnclipu.wi v2, v8, 0    # B8

vmv.v.v v0, v10

# Store the result with strided store (interleaving)
vsseg4e8.v v0, ({1})