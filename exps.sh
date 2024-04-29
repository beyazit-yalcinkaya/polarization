for gamma in 0.0 0.25 0.5 0.75 1.0
do
    for h in 0 1 2 4 8
    do
        for tolerance in 0 1 10 100 1000
        do
            for seed in {10..29}
            do
                python run.py $gamma $h $tolerance $seed &
                echo Running python run.py $gamma $h $tolerance $seed
            done
            wait
            echo Done with batch
        done
    done
done