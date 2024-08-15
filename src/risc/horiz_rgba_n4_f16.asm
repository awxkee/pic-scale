#  {0} - Weights ptr
#  {1} - Bounds Start
#  {2} - Bounds size
#  {3} - Source ptr
#  {4} - Dest ptr
#  {5} - x position
#  {6} - Src stride
#  {7} - Dst stride

    li t4, 4
    vsetvli t4, t4, e16, m1, ta, ma
    vxor.vv v1, v1, v1       # Vector register v1 = 0.0 for storing results
    vxor.vv v2, v2, v2       # Vector register v1 = 0.0 for storing results
    vxor.vv v3, v3, v3       # Vector register v1 = 0.0 for storing results
    vxor.vv v4, v4, v4       # Vector register v1 = 0.0 for storing results

    mv t5, {1}

    xor t2, t2, t2          # Loop counter = 0
    j 2f
1:
    flw ft1, 0({0})          # Load weight[j] into ft0 (scalar float)
    fcvt.h.s ft2, ft1        # Convert weight to half precision
    vfmv.v.f v9, ft2         # Broadcast weight[j] to all elements of v9

    slli t6, t5, 3           # Shift by px ( mul by 8 sizeof(f16) * 4 ] )
    add t1, {3}, t6
    add t3, t1, {6}
    add t4, t3, {6}
    add t6, t4, {6}

    vle16.v v5, (t1)         # Load 32-bit float into v5
    vle16.v v6, (t3)         # Load 32-bit float into v6
    vle16.v v7, (t4)         # Load 32-bit float into v7
    vle16.v v8, (t6)         # Load 32-bit float into v8

    vfmacc.vv v1, v9, v5   # FMA - store_0 = store_0 + (item_row * v_weight)
    vfmacc.vv v2, v9, v6   # FMA - store_1 = store_1 + (item_row * v_weight)
    vfmacc.vv v3, v9, v7   # FMA - store_2 = store_2 + (item_row * v_weight)
    vfmacc.vv v4, v9, v8   # FMA - store_3 = store_3 + (item_row * v_weight)

    addi {0}, {0}, 4
    addi t2, t2, 1
    addi t5, t5, 1
2:
    blt t2, {2}, 1b  # if j < bounds.size, exit loop
3:
    # Store the result in destination memory
    slli t1, {5}, 3         # t1 = cx * 4 * sizeof(f16) (assuming 2 bytes per half float)
    add {4}, {4}, t1        # dst_ptr = dst_ptr + px
    add t2, {4}, {7}        # dst_ptr = dst_ptr + px + dst_stride
    add t3, t2, {7}         # dst_ptr = dst_ptr + px + dst_stride * 2
    add t4, t3, {7}         # dst_ptr = dst_ptr + px + dst_stride * 3
    vse16.v v1, ({4})       # Store the result from v8[0] to dst_ptr
    vse16.v v2, (t2)       # Store the result from v8[0] to dst_ptr
    vse16.v v3, (t3)       # Store the result from v8[0] to dst_ptr
    vse16.v v4, (t4)       # Store the result from v8[0] to dst_ptr
