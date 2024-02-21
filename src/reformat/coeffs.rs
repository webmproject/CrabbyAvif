use crate::*;

impl ColorPrimaries {
    pub fn y_coeffs(&self) -> [f32; 3] {
        // TODO: implement.
        [0.0, 0.0, 0.0]
    }
}

fn calculate_yuv_coefficients_from_cicp(
    color_primaries: ColorPrimaries,
    matrix_coefficients: MatrixCoefficients,
) -> Option<[f32; 3]> {
    match matrix_coefficients {
        MatrixCoefficients::ChromaDerivedNcl => Some(color_primaries.y_coeffs()),
        _ => {
            let mut lookup = HashMap::with_hasher(NonRandomHasherState);
            lookup.insert(MatrixCoefficients::Bt709, (0.2126f32, 0.0722));
            lookup.insert(MatrixCoefficients::Fcc, (0.30, 0.11));
            lookup.insert(MatrixCoefficients::Bt470bg, (0.299, 0.114));
            lookup.insert(MatrixCoefficients::Bt601, (0.299, 0.114));
            lookup.insert(MatrixCoefficients::Smpte240, (0.212, 0.087));
            lookup.insert(MatrixCoefficients::Bt2020Ncl, (0.2627, 0.0593));
            lookup
                .get(&matrix_coefficients)
                .map(|x| [x.0, 1.0 - x.0 - x.1, x.1])
        }
    }
}

pub fn calculate_yuv_coefficients(
    color_primaries: ColorPrimaries,
    matrix_coefficients: MatrixCoefficients,
) -> [f32; 3] {
    // Return known coefficients or fall back to BT.601.
    calculate_yuv_coefficients_from_cicp(color_primaries, matrix_coefficients).unwrap_or(
        calculate_yuv_coefficients_from_cicp(color_primaries, MatrixCoefficients::Bt601).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv_coefficients() {
        fn assert_near(a: [f32; 3], b: [f32; 3]) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() <= std::f32::EPSILON);
            }
        }

        assert_near(
            calculate_yuv_coefficients(ColorPrimaries::Unknown, MatrixCoefficients::Bt601),
            [0.299f32, 0.587f32, 0.114f32], // Kr,Kg,Kb as https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
        );
        assert_near(
            calculate_yuv_coefficients(ColorPrimaries::Unknown, MatrixCoefficients::Unspecified),
            [0.299f32, 0.587f32, 0.114f32], // Falls back to Bt601.
        );
        assert_near(
            calculate_yuv_coefficients(ColorPrimaries::Unknown, MatrixCoefficients::Smpte240),
            [0.212f32, 1f32 - 0.212 - 0.087, 0.087f32], // Kr,Kg,Kb as https://en.wikipedia.org/wiki/YCbCr#SMPTE_240M_conversion
        );
    }
}
