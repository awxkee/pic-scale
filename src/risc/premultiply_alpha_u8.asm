li {t0}, -1
vsetvli {t0}, {t0}, e8, m1, ta, ma

li   {t4}, 127

vlseg4e8.v v0, ({0})

vmv.v.v v12, v3

# Widening lower parts
vwmulu.vv v4, v0, v3    # Low R16
vwmulu.vv v6, v1, v3    # Low G16
vwmulu.vv v8, v2, v3    # Low B16

li {t1}, -1
vsetvli {t1}, {t1}, e16, m1, ta, ma

# Div by 255

vsrl.vi v12, v4, 8    # Low R16
vadd.vx v13, v4, {t4}
vadd.vv v4, v12, v13
vsrl.vi v4, v4, 8

vsrl.vi v12, v5, 8    # High R16
vadd.vx v13, v5, {t4}
vadd.vv v5, v12, v13
vsrl.vi v5, v5, 8

vsrl.vi v12, v6, 8    # Low G16
vadd.vx v13, v6, {t4}
vadd.vv v6, v12, v13
vsrl.vi v6, v6, 8

vsrl.vi v12, v7, 8    # High G16
vadd.vx v13, v7, {t4}
vadd.vv v7, v12, v13
vsrl.vi v7, v7, 8

vsrl.vi v12, v8, 8    # Low B16
vadd.vx v13, v8, {t4}
vadd.vv v8, v12, v13
vsrl.vi v8, v8, 8

vsrl.vi v12, v9, 8    # Low B16
vadd.vx v13, v9, {t4}
vadd.vv v9, v12, v13
vsrl.vi v9, v9, 8

li {t0}, -1
vsetvli {t0}, {t0}, e8, m1, ta, ma

# Narrowing

vnclipu.wi v0, v4, 0    # R8
vnclipu.wi v1, v6, 0    # G8
vnclipu.wi v2, v8, 0    # B8

# Store the result with strided store (interleaving)
vsseg4e8.v v0, ({1})