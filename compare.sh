#! /bin/bash

dir="/Users/vigneshv/code/av1-avif/testFiles"

c_dir="/tmp/gold"
rust_dir="/tmp/actual"
mkdir -p ${c_dir}
mkdir -p ${rust_dir}

stats="/Users/vigneshv/code/rust_pg/rust-libavif/match_stats.csv"
echo -n "" > $stats

AVIFDEC="avifdec"
RUST_AVIFDEC="/Users/vigneshv/code/rust_pg/rust-libavif/target/debug/rust-libavif"

for file in $(find ${dir} -iname "*.avif");
do
    bname=$(basename ${file});
    no_extn=${bname%.*};
    c_out="${c_dir}/${no_extn}.y4m";
    rust_out="${rust_dir}/${no_extn}.y4m";

    echo -n "${no_extn},"

    # run C binary
    $AVIFDEC --no-strict --jobs 8 "${file}" "${c_out}" > /dev/null 2>&1
    c_run=$?
    echo -n "${c_run},"

    # run rust binary
    $RUST_AVIFDEC "${file}" "${rust_out}" --no-png > /dev/null 2>&1
    rust_run=$?
    echo "${rust_run}"

    matched="0"

    if [ ${c_run} -eq 0 ]; then
        if [ ${rust_run} -eq 0 ]; then
            cmp -s "${c_out}" "${rust_out}"
            cmp_res=$?
            if [ ${cmp_res} -eq 0 ]; then
                echo "matched";
                matched="1"
            else
                echo "mismatched";
            fi
        else
            echo "rust run failed";
        fi
    else
        echo "c run failed.";    
    fi

    echo "${no_extn},${c_run},${rust_run},${matched}" >> ${stats}
done

echo
echo "====="
echo
echo "mismatched files: "
grep -E "0$" match_stats.csv
echo
echo -n "mismatched file count: "
grep -E "0$" match_stats.csv | wc -l