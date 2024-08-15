    li t1, -1
    vsetvli t1, t1, e32, m1, ta, ma
    vxor.vv v1, v1, v1       # Vector register v1 = 0.0 for storing results

    xor t2, t2, t2           # Initialize loop counter j = 0

    j 2
1:
    add t3, {0}, t2           # t3 = py = start_y + j

    # Load weight from filter.add(j)
    slli t4, t2, 2           # t4 = j * 4 (assuming 4 bytes per float)
    add t4, {5}, t4          # t4 = address of weight[j]
    flw ft1, 0(t4)           # Load weight[j] into ft0 (scalar float)
    vfmv.v.f v2, ft1         # Broadcast weight[j] to all elements of v2

    # Calculate src_ptr = src + src_stride * py
    mul t6, {3}, t3          # t6 = src_stride * py
    slli t6, t6, 2           # Mul by 4 (assuming 4 bytes per float)
    slli t5, {1}, 2          # t5 = cx * 4 (assuming 4 bytes per float)
    add t6, t6, t5           # t6 = src_y + src_x
    add t3, {2}, t6          # t3 = src_ptr

    # Load source row data
    vle32.v v3, (t3)       # Load 32-bit float into v10

    vfmacc.vv v1, v2, v3   # FMA - store_0 = store_0 + (item_row * v_weight)

    addi t2, t2, 1
2:
    # Check if j < bounds.size
    blt t2, {6}, 1b  # if j < bounds.size, exit loop

    # Store the result in destination memory
    slli t1, {1}, 2         # t3 = cx * 4 (assuming 4 bytes per float)
    add t1, {4}, t1         # t8 = s_ptr = src_ptr + px
    vse32.v v1, (t1)        # Store the result from v8[0] to dst_ptr