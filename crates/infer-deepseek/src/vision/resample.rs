use image::RgbImage;

struct ResampleCoeffs {
    bounds: Vec<(usize, usize)>,
    coeffs_int: Vec<i32>,
    ksize: usize,
}

const PRECISION_BITS: i32 = 22;
const PRECISION_SCALE: f64 = (1 << PRECISION_BITS) as f64;
const ROUNDING_BIAS: i64 = 1 << (PRECISION_BITS - 1);

fn clip8(value: i64) -> u8 {
    let shifted = value >> PRECISION_BITS;
    shifted.clamp(0, 255) as u8
}

fn round_half_towards_zero(value: f64) -> isize {
    if value >= 0.0 {
        (value + 0.5).floor() as isize
    } else {
        (value + 0.5).ceil() as isize
    }
}

fn bicubic_kernel(value: f64) -> f64 {
    const A: f64 = -0.5;
    let x = value.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn compute_resample_coeffs(input_size: usize, output_size: usize) -> ResampleCoeffs {
    let scale = input_size as f64 / output_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;

    let mut bounds = Vec::with_capacity(output_size);
    let mut coeffs_int = vec![0i32; output_size * ksize];
    let mut coeff_row = vec![0.0f64; ksize];

    for out_index in 0..output_size {
        let center = (out_index as f64 + 0.5) * scale;
        let mut xmin = round_half_towards_zero(center - support);
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = round_half_towards_zero(center + support);
        if xmax > input_size as isize {
            xmax = input_size as isize;
        }
        if xmin >= input_size as isize {
            xmin = input_size.saturating_sub(1) as isize;
        }
        if xmax <= xmin {
            xmax = xmin + 1;
        }
        let length = (xmax - xmin) as usize;
        let ss = 1.0 / filterscale;
        coeff_row.fill(0.0);
        let mut sum = 0.0;
        for i in 0..length {
            let sample_pos = xmin as f64 + i as f64;
            let weight = bicubic_kernel((sample_pos - center + 0.5) * ss);
            coeff_row[i] = weight;
            sum += weight;
        }
        for i in length..ksize {
            coeff_row[i] = 0.0;
        }
        if sum != 0.0 {
            for i in 0..length {
                coeff_row[i] /= sum;
            }
        }
        let coeff_row_int = &mut coeffs_int[out_index * ksize..out_index * ksize + ksize];
        for i in 0..ksize {
            let v = coeff_row[i];
            coeff_row_int[i] = if v < 0.0 {
                (-0.5 + v * PRECISION_SCALE) as i32
            } else {
                (0.5 + v * PRECISION_SCALE) as i32
            };
        }
        bounds.push((xmin as usize, length));
    }

    ResampleCoeffs {
        bounds,
        coeffs_int,
        ksize,
    }
}

pub(crate) fn resize_bicubic(source: &RgbImage, width: u32, height: u32) -> RgbImage {
    let src_width = source.width() as usize;
    let src_height = source.height() as usize;
    let dst_width = width as usize;
    let dst_height = height as usize;

    if dst_width == 0 || dst_height == 0 {
        return RgbImage::new(width, height);
    }

    let coefficients_x = compute_resample_coeffs(src_width, dst_width);
    let coefficients_y = compute_resample_coeffs(src_height, dst_height);
    let ksize_x = coefficients_x.ksize;
    let ksize_y = coefficients_y.ksize;

    let mut horizontal = vec![0u8; src_height * dst_width * 3];
    let src_buffer = source.as_raw();

    for y in 0..src_height {
        let src_row_offset = y * src_width * 3;
        for dst_x in 0..dst_width {
            let (start, len) = coefficients_x.bounds[dst_x];
            let coeffs = &coefficients_x.coeffs_int[dst_x * ksize_x..dst_x * ksize_x + len];
            let mut acc = [ROUNDING_BIAS; 3];
            for (i, &weight) in coeffs.iter().enumerate() {
                let src_x = start + i;
                let pixel_offset = src_row_offset + src_x * 3;
                acc[0] += (src_buffer[pixel_offset] as i64) * (weight as i64);
                acc[1] += (src_buffer[pixel_offset + 1] as i64) * (weight as i64);
                acc[2] += (src_buffer[pixel_offset + 2] as i64) * (weight as i64);
            }
            let dst_offset = (y * dst_width + dst_x) * 3;
            horizontal[dst_offset] = clip8(acc[0]);
            horizontal[dst_offset + 1] = clip8(acc[1]);
            horizontal[dst_offset + 2] = clip8(acc[2]);
        }
    }

    let mut output = vec![0u8; dst_width * dst_height * 3];
    for dst_y in 0..dst_height {
        let (start, len) = coefficients_y.bounds[dst_y];
        let coeffs = &coefficients_y.coeffs_int[dst_y * ksize_y..dst_y * ksize_y + len];
        for dst_x in 0..dst_width {
            let mut acc = [ROUNDING_BIAS; 3];
            for (i, &weight) in coeffs.iter().enumerate() {
                let src_y = start + i;
                let src_offset = (src_y * dst_width + dst_x) * 3;
                acc[0] += (horizontal[src_offset] as i64) * (weight as i64);
                acc[1] += (horizontal[src_offset + 1] as i64) * (weight as i64);
                acc[2] += (horizontal[src_offset + 2] as i64) * (weight as i64);
            }
            let dst_offset = (dst_y * dst_width + dst_x) * 3;
            output[dst_offset] = clip8(acc[0]);
            output[dst_offset + 1] = clip8(acc[1]);
            output[dst_offset + 2] = clip8(acc[2]);
        }
    }

    RgbImage::from_raw(width, height, output).expect("invalid resized image dimensions")
}
