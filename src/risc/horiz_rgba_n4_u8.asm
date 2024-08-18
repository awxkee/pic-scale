#  {0} - Weights ptr
#  {1} - Bounds Start
#  {2} - Bounds size
#  {3} - Source ptr
#  {4} - Dest ptr
#  {5} - x position
#  {6} - Src stride
#  {7} - Dst stride

    li {t4}, 4
    vsetvli {t4}, {t4}, e32, m1, ta, ma

    li {t2}, 0x8000

    vmv.v.x v2, {t2}           # Vector register v1 = INITIAL_ROUNDING for storing results
    vmv.v.x v4, {t2}           # Vector register v1 = INITIAL_ROUNDING for storing results
    vmv.v.x v6, {t2}           # Vector register v1 = INITIAL_ROUNDING for storing results
    vmv.v.x v8, {t2}           # Vector register v1 = INITIAL_ROUNDING for storing results

    mv {t5}, {1}

    xor {t2}, {t2}, {t2}          # Loop counter = 0
    j 2f
1:
    li {t4}, 4
    vsetvli zero, {t4}, e8, m1, ta, ma

    slli {t6}, {t5}, 2            # Shift by px ( mul by 4 [ sizeof(u8) * 4 ] )
    add {t6}, {3}, {t6}
    vle8.v v3, ({t6})           # Load 8-bit u8 into v3
    vwaddu.vx v10, v3, x0     # Pixel 16

    add {t6}, {t6}, {6}
    vle8.v v3, ({t6})           # Load 8-bit u8 into v3
    vwaddu.vx v12, v3, x0     # Pixel 16

    add {t6}, {t6}, {6}
    vle8.v v3, ({t6})           # Load 8-bit u8 into v3
    vwaddu.vx v14, v3, x0     # Pixel 16

    add {t6}, {t6}, {6}
    vle8.v v3, ({t6})           # Load 8-bit u8 into v3
    vwaddu.vx v16, v3, x0     # Pixel 16

    li {t4}, 4
    vsetvli {t4}, {t4}, e16, m1, ta, ma

    lh {t1}, 0({0})           # Load weight[j] into ft0 (scalar uint16)
    vmv.v.x v18, {t1}         # Broadcast weight[j] to all elements of v2

    vwmacc.vv v2, v10, v18   # FMA - store_0 = store_0 + (item_row * v_weight)
    vwmacc.vv v4, v12, v18   # FMA - store_1 = store_1 + (item_row * v_weight)
    vwmacc.vv v6, v14, v18   # FMA - store_2 = store_2 + (item_row * v_weight)
    vwmacc.vv v8, v16, v18   # FMA - store_3 = store_3 + (item_row * v_weight)

    addi {0}, {0}, 2
    addi {t2}, {t2}, 1
    addi {t5}, {t5}, 1
2:
    blt {t2}, {2}, 1b  # if j < bounds.size, exit loop
3:
    li {t4}, 4
    li {t2}, 255
    vsetvli {t4}, {t4}, e16, m1, ta, ma

    vnclip.wi v10, v2, 15   # Saturate signed by >> 15 to e16
    vnclip.wi v12, v4, 15   # Saturate signed by >> 15 to e16
    vnclip.wi v14, v6, 15   # Saturate signed by >> 15 to e16
    vnclip.wi v16, v8, 15   # Saturate signed by >> 15 to e16

    vmax.vx v10, v10, x0
    vmin.vx v10, v10, {t2}
    vmax.vx v12, v12, x0
    vmin.vx v12, v12, {t2}
    vmax.vx v14, v14, x0
    vmin.vx v14, v14, {t2}
    vmax.vx v16, v16, x0
    vmin.vx v16, v16, {t2}

    li {t1}, 4
    vsetvli zero, {t1}, e8, m1, ta, ma

    vnclipu.wi v1, v10, 0  # u8 row
    vnclipu.wi v2, v12, 0  # u8 row
    vnclipu.wi v3, v14, 0  # u8 row
    vnclipu.wi v4, v16, 0  # u8 row

    # Store the result in destination memory
    slli {t1}, {5}, 2         # {t1} = cx * 4 * sizeof(u8)
    add {4}, {4}, {t1}        # t8 = dst_ptr = dst_ptr + px
    vse8.v v1, ({4})        # Store the result from v8[0] to dst_ptr

    add {4}, {4}, {7}        # t8 = dst_ptr = dst_ptr + px
    vse8.v v2, ({4})        # Store the result from v8[0] to dst_ptr

    add {4}, {4}, {7}        # t8 = dst_ptr = dst_ptr + px
    vse8.v v3, ({4})        # Store the result from v8[0] to dst_ptr

    add {4}, {4}, {7}        # t8 = dst_ptr = dst_ptr + px
    vse8.v v4, ({4})        # Store the result from v8[0] to dst_ptr

