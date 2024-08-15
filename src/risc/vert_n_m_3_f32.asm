    li t4, -1
    vsetvli t4, t4, e32, m1, ta, ma # t4 - stores max vec length
    slli t4, t4, 2           # Adjust vector shift by N*4 bytes
    vxor.vv v1, v1, v1       # Vector register v1 = 0.0 for storing results
    vxor.vv v2, v2, v2       # Vector register v2 = 0.0 for storing results
    vxor.vv v3, v3, v3       # Vector register v2 = 0.0 for storing results

    xor t2, t2, t2           # Initialize loop counter j = 0

    j 2
1:
    add t3, {0}, t2           # t3 = py = start_y + j

    # Load weight from filter.add(j)
    slli t5, t2, 2           # t5 = j * 4 (assuming 4 bytes per float)
    add t5, {5}, t5          # t5 = address of weight[j]
    flw ft1, 0(t5)           # Load weight[j] into ft0 (scalar float)
    vfmv.v.f v4, ft1         # Broadcast weight[j] to all elements of v3

    # Calculate src_ptr = src + src_stride * py
    mul t6, {3}, t3          # t6 = src_stride * py
    slli t6, t6, 2           # Mul by 4 (assuming 4 bytes per float)
    slli t5, {1}, 2          # t5 = cx * 4 (assuming 4 bytes per float)

    add t6, t6, t5           # t6 = src_y + src_x
    add t3, {2}, t6          # t3 = src_ptr
    add t5, t3, t4           # Next items
    add t6, t5, t4           # Doubled next items

    # Load source row data
    vle32.v v5, (t3)       # Load 32-bit float into v10
    vle32.v v6, (t5)
    vle32.v v7, (t6)

    vfmacc.vv v1, v4, v5   # FMA - store_0 = store_0 + (item_row0 * v_weight)
    vfmacc.vv v2, v4, v6   # FMA - store_1 = store_1 + (item_row1 * v_weight)
    vfmacc.vv v3, v4, v7   # FMA - store_2 = store_2 + (item_row2 * v_weight)

    addi t2, t2, 1
2:
    # Check if j < bounds.size
    blt t2, {6}, 1b  # if j < bounds.size, exit loop

    # Store the result in destination memory
    slli t1, {1}, 2         # t3 = cx * 4 (assuming 4 bytes per float)
    add t1, {4}, t1         # t8 = s_ptr = src_ptr + px
    add t2, t1, t4
    add t3, t2, t4
    vse32.v v1, (t1)        # Store the result from v1[0] to dst_ptr
    vse32.v v2, (t2)        # Store the result from v2[0] to dst_ptr + N
    vse32.v v3, (t3)        # Store the result from v3[0] to dst_ptr + N * 2