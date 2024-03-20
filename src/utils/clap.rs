use crate::internal_utils::*;
use crate::*;

#[derive(Debug, Clone, Copy)]
pub struct CleanAperture {
    pub width: Fraction,
    pub height: Fraction,
    pub horiz_off: Fraction,
    pub vert_off: Fraction,
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropRect {
    fn is_valid(&self, image_width: u32, image_height: u32, pixel_format: PixelFormat) -> bool {
        if self.width == 0
            || self.height == 0
            || self.x.checked_add(self.width).is_none()
            || self.y.checked_add(self.height).is_none()
            || self.x + self.width > image_width
            || self.y + self.height > image_height
        {
            return false;
        }
        match pixel_format {
            PixelFormat::Yuv420 => self.x % 2 == 0 && self.y % 2 == 0,
            PixelFormat::Yuv422 => self.x % 2 == 0,
            _ => true,
        }
    }

    pub fn create_from(
        clap: &CleanAperture,
        image_width: u32,
        image_height: u32,
        pixel_format: PixelFormat,
    ) -> AvifResult<Self> {
        if clap.width.d == 0
            || clap.height.d == 0
            || clap.horiz_off.d == 0
            || clap.vert_off.d == 0
            || clap.width.is_negative
            || clap.height.is_negative
            || !clap.width.is_integer()
            || !clap.height.is_integer()
        {
            println!("invalid clap");
            return Err(AvifError::UnknownError);
        }
        let clap_width = clap.width.get_u32()?;
        let clap_height = clap.height.get_u32()?;
        let crop_x = Fraction::new(image_width, 2)
            .add(clap.horiz_off)?
            .sub(Fraction::new(clap_width, 2))?;
        let crop_y = Fraction::new(image_height, 2)
            .add(clap.vert_off)?
            .sub(Fraction::new(clap_height, 2))?;
        if !crop_x.is_integer() || !crop_y.is_integer() || crop_x.is_negative || crop_y.is_negative
        {
            return Err(AvifError::UnknownError);
        }
        let rect = CropRect {
            x: crop_x.get_u32()?,
            y: crop_y.get_u32()?,
            width: clap_width,
            height: clap_height,
        };
        if rect.is_valid(image_width, image_height, pixel_format) {
            Ok(rect)
        } else {
            Err(AvifError::UnknownError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestParam {
        image_width: u32,
        image_height: u32,
        pixel_format: PixelFormat,
        clap: CleanAperture,
        rect: Option<CropRect>,
    }

    macro_rules! valid {
        ($a: expr, $b: expr, $c: ident, $d: expr, $e: expr, $f: expr, $g: expr, $h: expr, $i: expr,
         $j: expr, $k: expr, $l: expr, $m: expr, $n: expr, $o: expr) => {
            TestParam {
                image_width: $a,
                image_height: $b,
                pixel_format: PixelFormat::$c,
                clap: CleanAperture {
                    width: Fraction::new_i32($d, $e),
                    height: Fraction::new_i32($f, $g),
                    horiz_off: Fraction::new_i32($h, $i),
                    vert_off: Fraction::new_i32($j, $k),
                },
                rect: Some(CropRect {
                    x: $l,
                    y: $m,
                    width: $n,
                    height: $o,
                }),
            }
        };
    }

    macro_rules! invalid {
        ($a: expr, $b: expr, $c: ident, $d: expr, $e: expr, $f: expr, $g: expr, $h: expr, $i: expr,
         $j: expr, $k: expr) => {
            TestParam {
                image_width: $a,
                image_height: $b,
                pixel_format: PixelFormat::$c,
                clap: CleanAperture {
                    width: Fraction::new_i32($d, $e),
                    height: Fraction::new_i32($f, $g),
                    horiz_off: Fraction::new_i32($h, $i),
                    vert_off: Fraction::new_i32($j, $k),
                },
                rect: None,
            }
        };
    }

    #[rustfmt::skip]
    const TEST_PARAMS: [TestParam; 20] = [
        valid!(120, 160, Yuv420, 96, 1, 132, 1, 0, 1, 0, 1, 12, 14, 96, 132),
        valid!(120, 160, Yuv420, 60, 1, 80, 1, -30, 1, -40, 1, 0, 0, 60, 80),
        valid!(100, 100, Yuv420, 99, 1, 99, 1, -1, 2, -1, 2, 0, 0, 99, 99),
        invalid!(120, 160, Yuv420, 96, 0, 132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, -96, 1, 132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 132, 0, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, -132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 132, 1, 0, 0, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 132, 1, -1, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 132, 1, 0, 1, 0, 0),
        invalid!(120, 160, Yuv420, 96, 1, 132, 1, 0, 1, -1, 1),
        invalid!(120, 160, Yuv420, -96, 1, 132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 0, 1, 132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, -132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 0, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 5, 132, 1, 0, 1, 0, 1),
        invalid!(120, 160, Yuv420, 96, 1, 132, 5, 0, 1, 0, 1),
        invalid!(722, 1024, Yuv420, 385, 1, 330, 1, 103, 1, -308, 1),
        invalid!(1024, 722, Yuv420, 330, 1, 385, 1, -308, 1, 103, 1),
        invalid!(99, 99, Yuv420, 99, 1, 99, 1, -1, 2, -1, 2),
    ];

    #[test_case::test_matrix(0usize..20)]
    fn valid_clap_to_rect(index: usize) {
        let param = &TEST_PARAMS[index];
        let rect = CropRect::create_from(
            &param.clap,
            param.image_width,
            param.image_height,
            param.pixel_format,
        );
        if param.rect.is_some() {
            assert!(rect.is_ok());
            let rect = rect.unwrap();
            let expected_rect = param.rect.unwrap_ref();
            assert_eq!(rect.x, expected_rect.x);
            assert_eq!(rect.y, expected_rect.y);
            assert_eq!(rect.width, expected_rect.width);
            assert_eq!(rect.height, expected_rect.height);
        } else {
            assert!(rect.is_err());
        }
    }
}
