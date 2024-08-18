    li {t1}, -1
    vsetvli {t1}, {t1}, e32, m1, ta, ma

    li {t2}, 0x8000

    vmv.v.x v2, {t2}           # Vector register v1 = INITIAL_ROUNDING for storing results
    vmv.v.x v3, {t2}           # Vector register v2 = INITIAL_ROUNDING for storing results
    vmv.v.x v4, {t2}           # Vector register v3 = INITIAL_ROUNDING for storing results
    vmv.v.x v5, {t2}           # Vector register v4 = INITIAL_ROUNDING for storing results

    li {t1}, -1
    vsetvli {t1}, {t1}, e8, m1, ta, ma

    xor {t2}, {t2}, {t2}           # Initialize loop counter j = 0

    j 2
1:
    add {t3}, {0}, {t2}           # {t3} = py = start_y + j

    # Calculate src_ptr = src + src_stride * py
    mul {t6}, {3}, {t3}          # {t6} = src_stride * py
    add {t6}, {t6}, {1}          # {t6} = src_y + src_x
    add {t3}, {2}, {t6}          # {t3} = src_ptr

    li {t1}, -1
    vsetvli zero, {t1}, e8, m1, ta, ma

    # Load source row data
    vle8.v v7, ({t3})         # Load 16-bit uin{t1}6 into v10
    vwaddu.vx v8, v7, x0    # Pixel 16

    li {t1}, -1
    vsetvli zero, {t1}, e16, m1, ta, ma

    # Load weight from filter.add(j)
    slli {t4}, {t2}, 1           # {t4} = j * 2 (assuming 2 bytes per uint16)
    add {t4}, {5}, {t4}          # {t4} = address of weight[j]
    lh {t1}, 0({t4})            # Load weight[j] into ft0 (scalar uint16)
    vmv.v.x v10, {t1}          # Broadcast weight[j] to all elements of v2

    vwmacc.vv v2, v8, v10   # FMA - store_0/store_1 = store_0/store_1 + (item_row * v_weight)
    vwmacc.vv v4, v9, v10   # FMA - store_2/store_3 = store_2/store_3 + (item_row * v_weight)

    addi {t2}, {t2}, 1
2:
    # Check if j < bounds.size
    blt {t2}, {6}, 1b  # if j < bounds.size, exit loop

    li {t2}, 255

    li {t1}, -1
    vsetvli zero, {t1}, e16, m1, ta, ma

    vnclip.wi v10, v2, 15   # Saturate signed by >> 15 to e16
    vnclip.wi v11, v4, 15   # Saturate signed by >> 15 to e16

    vmax.vx v10, v10, x0
    vmax.vx v11, v11, x0
    vmin.vx v10, v10, {t2}
    vmin.vx v11, v11, {t2}

    li {t1}, -1
    vsetvli zero, {t1}, e8, m1, ta, ma

    vnclipu.wi v1, v10, 0  # u8 row

    # Store the result in destination memory
    add {t1}, {4}, {1}        # t8 = s_ptr = src_ptr + px
    vse8.v v1, ({t1})         # Store the result from v8[0] to dst_ptr