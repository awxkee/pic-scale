#  {0} - Weights ptr
#  {1} - Bounds Start
#  {2} - Bounds size
#  {3} - Source ptr
#  {4} - Dest ptr
#  {5} - x position

    li t4, 4
    vsetvli t4, t4, e32, m1, ta, ma
    vxor.vv v1, v1, v1       # Vector register v1 = 0.0 for storing results

    mv t5, {1}

    xor t2, t2, t2          # Loop counter = 0
    j 2f
1:
    flw ft1, 0({0})           # Load weight[j] into ft0 (scalar float)
    vfmv.v.f v2, ft1         # Broadcast weight[j] to all elements of v2

    slli t6, t5, 4           # Shift by px ( mul by 16 [ sizeof(f32) * 4 ] )
    add t6, {3}, t6
    vle32.v v3, (t6)         # Load 32-bit float into v10

    vfmacc.vv v1, v2, v3   # FMA - store_0 = store_0 + (item_row * v_weight)

    addi {0}, {0}, 4
    addi t2, t2, 1
    addi t5, t5, 1
2:
    blt t2, {2}, 1b  # if j < bounds.size, exit loop
3:
    # Store the result in destination memory
    slli t1, {5}, 4         # t1 = cx * 4 * sizeof(f32) (assuming 4 bytes per float)
    add {4}, {4}, t1        # t8 = dst_ptr = dst_ptr + px
    vse32.v v1, ({4})        # Store the result from v8[0] to dst_ptr
