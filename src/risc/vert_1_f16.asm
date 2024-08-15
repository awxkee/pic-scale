    xor t2, t2, t2           # Initialize loop counter j = 0
    fmv.h.x ft1, t2

    j 2f
1:
    add t3, {0}, t2           # t3 = py = start_y + j

    # Load weight from filter.add(j)
    slli t4, t2, 2           # t4 = j * 4 (assuming 4 bytes per half float)
    add t5, {5}, t4          # t5 = address of weight[j]
    flw ft3, 0(t5)           # Load weight[j] into ft3
    fcvt.h.s ft2, ft3        # Convert weight to half precision

    # Calculate src_ptr = src + src_stride * py
    mul t6, {3}, t3          # t6 = src_stride * py
    slli t6, t6, 1           # Mul by 2 (assuming 2 bytes per half float)
    slli t5, {1}, 1          # t5 = cx * 2 (assuming 2 bytes per half float)
    add t6, t6, t5           # t6 = src_y + src_x
    add t3, {2}, t6          # t7 = src_ptr

    # Load source row data
    flh ft3, 0(t3)           # Load src ptr into register

    fmadd.h ft1, ft2, ft3, ft1

    addi t2, t2, 1
2:
    # Check if j < bounds.size
    blt t2, {6}, 1b  # if j < bounds.size, exit loop

    # Store the result in destination memory
    slli t1, {1}, 1         # t3 = cx * 4 (assuming 4 bytes per float)
    add t1, {4}, t1         # t8 = s_ptr = src_ptr + px

    # Store the value in t0 into the memory location
    fsh ft1, 0(t1)