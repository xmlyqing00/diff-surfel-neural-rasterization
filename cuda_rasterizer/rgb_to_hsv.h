#include <torch/extension.h>



__device__ void hsv_to_rgb(float h, float s, float v, float &r, float &g, float &b) {
    int i;
    float f, p, q, t;
    if (s == 0) {
        r = g = b = v;
        return;
    }
    h /= 60;
    i = (int)h;
    f = h - i;
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

__device void hsv_to_rgb_backward(
    float h, float s, float v, float dL_dr, float dL_dg, float dL_db,
    float &dL_dh, float &dL_ds, float &dL_dv
) {
    int i;
    float f, p, q, t;
    if (s == 0) {
        dL_dv = dL_dr + dL_dg + dL_db;
        dL_ds = 0;
        dL_dh = 0;
        return;
    }
    h /= 60;
    i = (int)h;
    f = h - i;
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));
    switch (i) {
        case 0: dL_dv = dL_dr; dL_ds = dL_db; dL_dh = dL_dg; break;
        case 1: dL_dv = dL_db; dL_ds = dL_dr; dL_dh = dL_dg; break;
        case 2: dL_dv = dL_db; dL_ds = dL_dg; dL_dh = dL_dr; break;
        case 3: dL_dv = dL_dg; dL_ds = dL_db; dL_dh = dL_dr; break;
        case 4: dL_dv = dL_dg; dL_ds = dL_dr; dL_dh = dL_db; break;
        default: dL_dv = dL_dr; dL_ds = dL_dg; dL_dh = dL_db; break;
    }
    dL_dv += dL_dr + dL_dg;
    dL_ds *= -s;
    dL_dh *= -s * (1 - f);
    dL_ds += dL_dh * (f - 1);
    dL_dh *= v;
    dL_ds *= v;
    dL_dv *= 1 - s;
    
}