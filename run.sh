# questiones=("q1" "q2" "q3")
question="q4"
testcases=("cxover" "mutation" "pop")
num_gen=20000
cx_prob=0.8
mut_prob=0.04
pop_size=1000

# for question in "${questiones[@]}"; do
for testcase in "${testcases[@]}"; do
    python run.py \
        --question $question \
        --test $testcase \
        --num_gen $num_gen \
        --cx_prob $cx_prob \
        --mut_prob $mut_prob \
        --pop_size $pop_size
done
# done
