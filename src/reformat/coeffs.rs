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
    calculate_yuv_coefficients_from_cicp(color_primaries, matrix_coefficients)
        .unwrap_or([0.299, 0.114, 0.587])
}
