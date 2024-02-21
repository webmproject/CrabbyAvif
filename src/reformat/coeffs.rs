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
    let expand_coeffs = |y, v| Some([y, 1.0 - y - v, v]);
    match matrix_coefficients {
        MatrixCoefficients::ChromaDerivedNcl => Some(color_primaries.y_coeffs()),
        MatrixCoefficients::Bt709 => expand_coeffs(0.2126f32, 0.0722),
        MatrixCoefficients::Fcc => expand_coeffs(0.30, 0.11),
        MatrixCoefficients::Bt470bg | MatrixCoefficients::Bt601 => expand_coeffs(0.299, 0.114),
        MatrixCoefficients::Smpte240 => expand_coeffs(0.212, 0.087),
        MatrixCoefficients::Bt2020Ncl => expand_coeffs(0.2627, 0.0593),
        _ => None,
    }
}

pub fn calculate_yuv_coefficients(
    color_primaries: ColorPrimaries,
    matrix_coefficients: MatrixCoefficients,
) -> [f32; 3] {
    calculate_yuv_coefficients_from_cicp(color_primaries, matrix_coefficients)
        .unwrap_or([0.299, 0.114, 0.587])
}
