sep="####################################"

echo "Building..."
make all

echo $sep
echo $sep

echo "Part-A: CUDA Matrix Operations"
echo "Problem-1: Vector add and coalescing memory access"
values=(500 1000 2000)
for k in "${values[@]}"; do
    echo ""
    echo "vecadd00 $k"
    ./vecadd00 $k

    echo ""
    echo "vecadd01 $k"
    ./vecadd01 $k
    echo ""
done

echo $sep
echo $sep

echo "Problem-2: Shared CUDA Matrix Multiply"

echo "FOOTPRINT_SIZE=16"
values=(16 32 64)
for k in "${values[@]}"; do
    echo ""
    echo "matmult00 $k"
    ./matmult00 $k
    echo ""
done

echo $sep

echo "FOOTPRINT_SIZE=32"
values=(8 16 32)
for k in "${values[@]}"; do
    echo ""
    echo "matmult01 $k"
    ./matmult01 $k
    echo ""
done

echo $sep
echo $sep

echo "Part-B: CUDA Unified Memory"
um_args=("host" "device st" "device mt" "device mbmt")
um_ks=(1 5 10 50 100)
for k in "${um_ks[@]}"; do
    for args in "${um_args[@]}"; do
        echo ""
        echo "arrayadd $k $args"
        ./arrayadd $k $args

        echo ""
        echo "arrayaddUnifiedMemory $k $args"
        ./arrayaddUnifiedMemory $k $args
        echo ""
    done
done

echo $sep
echo $sep


echo "Part-C: Convolution in CUDA"
./conv all

echo $sep
echo $sep
