#!/bin/bash

dir="/Users/vigneshv/code/av1-avif/testFiles"
subdir="Link-U"
tmpfile="/tmp/avifdec_output.txt"

(
    cd "${dir}"
    for file in $(find ${subdir} -iname "*.avif" | sort);
    do
        full_file="${dir}/${file}";
        avifdec --no-strict --info "${full_file}" > ${tmpfile} 2>&1
        width=$(grep "Resolution" ${tmpfile} | cut -f2 -d: | cut -f1 -dx)
        height=$(grep "Resolution" ${tmpfile} | cut -f2 -d: | cut -f2 -dx)
        depth=$(grep "Bit Depth" ${tmpfile} | cut -f2 -d:)

        alpha_str=$(grep "Alpha" ${tmpfile} | cut -f2 -d:)
        if [ "${alpha_str}" == " Absent" ]; then
            alpha_present="false";
        else
            alpha_present="true";
        fi

        format_str=$(grep "* Format" ${tmpfile} | cut -f2 -d:)
        if [ "${format_str}" == " YUV420" ]; then
            yuv_format="PixelFormat::Yuv420";
        elif [ "${format_str}" == " YUV422" ]; then
            yuv_format="PixelFormat::Yuv422";
        elif [ "${format_str}" == " YUV444" ]; then
            yuv_format="PixelFormat::Yuv444";
        else
            yuv_format="PixelFormat::Monochrome";
        fi

        range_str=$(grep "* Range" ${tmpfile} | cut -f2 -d:)
        if [ "${range_str}" == " Full" ]; then
            full_range="true";
        else
            full_range="false";
        fi

        color_primaries=$(grep "Color Primaries" ${tmpfile} | cut -f2 -d:)
        transfer_characteristics=$(grep "Transfer Char" ${tmpfile} | cut -f2 -d:)
        matrix_coefficients=$(grep "Matrix Coeffs" ${tmpfile} | cut -f2 -d:)

        echo "ExpectedAvifImageInfo {"
        echo "filename: \"${file}\",";
        echo "width: ${width},"
        echo "height: ${height},"
        echo "depth: ${depth}",
        echo "yuv_format: ${yuv_format}",
        echo "alpha_present: ${alpha_present}",
        echo "full_range: ${full_range}",
        echo "color_primaries: ${color_primaries}",
        echo "transfer_characteristics: ${transfer_characteristics}",
        echo "matrix_coefficients: ${matrix_coefficients}",
        echo "},";
    done
)