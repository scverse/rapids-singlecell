#pragma once

// Exact orient2d/incircle from Shewchuk's predicates.c (public domain), as
// vendored in CuPy's Delaunay kernels (MIT). __dmul_rn blocks FMA contraction.
namespace spatial_predicates {

__device__ __constant__ double constants[] = {134217729.0};  // 2^27 + 1

#define MUL(a, b) __dmul_rn(a, b)

#define Fast_Two_Sum(a, b, x, y) \
    x = a + b;                   \
    bvirt = x - a;               \
    y = b - bvirt

#define Two_Sum(a, b, x, y) \
    x = a + b;              \
    bvirt = x - a;          \
    avirt = x - bvirt;      \
    bround = b - bvirt;     \
    around = a - avirt;     \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = a - b;               \
    bvirt = a - x;           \
    avirt = x + bvirt;       \
    bround = bvirt - b;      \
    around = a - avirt;      \
    y = around + bround

#define Split(a, ahi, alo)     \
    c = MUL(predConsts[0], a); \
    abig = c - a;              \
    ahi = c - abig;            \
    alo = a - ahi

#define Two_Product(a, b, x, y)  \
    x = MUL(a, b);               \
    Split(a, ahi, alo);          \
    Split(b, bhi, blo);          \
    err1 = x - MUL(ahi, bhi);    \
    err2 = err1 - MUL(alo, bhi); \
    err3 = err2 - MUL(ahi, blo); \
    y = MUL(alo, blo) - err3

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = MUL(a, b);                                 \
    Split(a, ahi, alo);                            \
    err1 = x - MUL(ahi, bhi);                      \
    err2 = err1 - MUL(alo, bhi);                   \
    err3 = err2 - MUL(ahi, blo);                   \
    y = MUL(alo, blo) - err3

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b, _i, x0);                \
    Two_Sum(a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0);            \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

__device__ int d_fast_expansion_sum_zeroelim(int elen, double* e, int flen,
                                             double* f, double* h) {
    double Q, Qnew, hh, bvirt, avirt, bround, around;
    double enow = e[0], fnow = f[0];
    int eindex = 0, findex = 0, hindex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        if (++eindex < elen) enow = e[eindex];
    } else {
        Q = fnow;
        if (++findex < flen) fnow = f[findex];
    }
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            if (++eindex < elen) enow = e[eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            if (++findex < flen) fnow = f[findex];
        }
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                if (++eindex < elen) enow = e[eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                if (++findex < flen) fnow = f[findex];
            }
            Q = Qnew;
            if (hh != 0.0) h[hindex++] = hh;
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
        Q = Qnew;
        if (hh != 0.0) h[hindex++] = hh;
    }
    if ((Q != 0.0) || (hindex == 0)) h[hindex++] = Q;
    return hindex;
}

// Same merge, keeping only the most significant nonzero component.
__device__ double d_fast_expansion_sum_sign(int elen, double* e, int flen,
                                            double* f) {
    double Q, lastTerm = 0.0, Qnew, hh, bvirt, avirt, bround, around;
    double enow = e[0], fnow = f[0];
    int eindex = 0, findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        if (++eindex < elen) enow = e[eindex];
    } else {
        Q = fnow;
        if (++findex < flen) fnow = f[findex];
    }
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            if (++eindex < elen) enow = e[eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            if (++findex < flen) fnow = f[findex];
        }
        Q = Qnew;
        if (hh != 0.0) lastTerm = hh;
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                if (++eindex < elen) enow = e[eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                if (++findex < flen) fnow = f[findex];
            }
            Q = Qnew;
            if (hh != 0.0) lastTerm = hh;
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
        Q = Qnew;
        if (hh != 0.0) lastTerm = hh;
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
        Q = Qnew;
        if (hh != 0.0) lastTerm = hh;
    }
    if (Q != 0.0) lastTerm = Q;
    return lastTerm;
}

__device__ int d_scale_twice_expansion_zeroelim(const double* predConsts,
                                                int elen, double* e, double b1,
                                                double b2, double* h) {
    double Q, sum, Q2, sum2, hh, product1, product2, product0, enow;
    double bvirt, avirt, bround, around, c, abig, ahi, alo, err1, err2, err3;
    double b1hi, b1lo, b2hi, b2lo;
    int hindex = 0;
    Split(b1, b1hi, b1lo);
    Split(b2, b2hi, b2lo);
    Two_Product_Presplit(e[0], b1, b1hi, b1lo, Q, hh);
    Two_Product_Presplit(hh, b2, b2hi, b2lo, Q2, hh);
    if (hh != 0) h[hindex++] = hh;
    for (int eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b1, b1hi, b1lo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) h[hindex++] = hh;
        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) h[hindex++] = hh;
        Fast_Two_Sum(product1, sum, Q, hh);
        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) h[hindex++] = hh;
        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) h[hindex++] = hh;
    }
    if (Q != 0) {
        Two_Product_Presplit(Q, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) h[hindex++] = hh;
        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) h[hindex++] = hh;
    }
    if ((Q2 != 0) || (hindex == 0)) h[hindex++] = Q2;
    return hindex;
}

__noinline__ __device__ double orient2dExact(const double* predConsts,
                                             const double* pa, const double* pb,
                                             const double* pc) {
    double axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
    double axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
    double aterms[4], bterms[4], cterms[4], v[8];
    double bvirt, avirt, bround, around, c, abig, ahi, alo, bhi, blo;
    double err1, err2, err3, _i, _j, _0;
    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pa[0], pc[1], axcy1, axcy0);
    Two_Two_Diff(axby1, axby0, axcy1, axcy0, aterms[3], aterms[2], aterms[1],
                 aterms[0]);
    Two_Product(pb[0], pc[1], bxcy1, bxcy0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0, bterms[3], bterms[2], bterms[1],
                 bterms[0]);
    Two_Product(pc[0], pa[1], cxay1, cxay0);
    Two_Product(pc[0], pb[1], cxby1, cxby0);
    Two_Two_Diff(cxay1, cxay0, cxby1, cxby0, cterms[3], cterms[2], cterms[1],
                 cterms[0]);
    int vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);
    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

// ab = pa.x * pb.y - pb.x * pa.y as a 4-term expansion.
__forceinline__ __device__ void two_mult_sub(const double* predConsts,
                                             const double* pa, const double* pb,
                                             double* ab) {
    double axby1, axby0, bxay1, bxay0;
    double bvirt, avirt, bround, around, c, abig, ahi, alo, bhi, blo;
    double err1, err2, err3, _i, _j, _0;
    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);
}

// ret = (a + b + c) * (fx0 * fx1 + fy0 * fy1), one incircle cofactor term.
__noinline__ __device__ int calc_det(const double* predConsts, double* a,
                                     double* b, double* c, double fx0,
                                     double fx1, double fy0, double fy1,
                                     double* temp2, double* temp3, double* detx,
                                     double* dety, double* ret) {
    int temp2len = d_fast_expansion_sum_zeroelim(4, a, 4, b, temp2);
    int temp3len = d_fast_expansion_sum_zeroelim(temp2len, temp2, 4, c, temp3);
    int xlen = d_scale_twice_expansion_zeroelim(predConsts, temp3len, temp3,
                                                fx0, fx1, detx);
    int ylen = d_scale_twice_expansion_zeroelim(predConsts, temp3len, temp3,
                                                fy0, fy1, dety);
    return d_fast_expansion_sum_zeroelim(xlen, detx, ylen, dety, ret);
}

__noinline__ __device__ double incircleExact(const double* predConsts,
                                             const double* pa, const double* pb,
                                             const double* pc,
                                             const double* pd) {
    double ab[4], bc[4], cd[4], da[4], ac[4], bd[4], temp8[8], temp12[12];
    double det48x[48], det48y[48], bdet[96], cdet[96], bcdet[192], addet[192];
    double *adet = bdet, *ddet = cdet;  // reused once bcdet is formed
    two_mult_sub(predConsts, pa, pb, ab);
    two_mult_sub(predConsts, pb, pc, bc);
    two_mult_sub(predConsts, pc, pd, cd);
    two_mult_sub(predConsts, pd, pa, da);
    two_mult_sub(predConsts, pa, pc, ac);
    two_mult_sub(predConsts, pb, pd, bd);
    int blen = calc_det(predConsts, cd, da, ac, pb[0], -pb[0], pb[1], -pb[1],
                        temp8, temp12, det48x, det48y, bdet);
    int clen = calc_det(predConsts, da, ab, bd, pc[0], pc[0], pc[1], pc[1],
                        temp8, temp12, det48x, det48y, cdet);
    int bclen = d_fast_expansion_sum_zeroelim(blen, bdet, clen, cdet, bcdet);
    for (int i = 0; i < 4; i++) {
        bd[i] = -bd[i];
        ac[i] = -ac[i];
    }
    int dlen = calc_det(predConsts, ab, bc, ac, pd[0], -pd[0], pd[1], -pd[1],
                        temp8, temp12, det48x, det48y, ddet);
    int alen = calc_det(predConsts, bc, cd, bd, pa[0], pa[0], pa[1], pa[1],
                        temp8, temp12, det48x, det48y, adet);
    int adlen = d_fast_expansion_sum_zeroelim(alen, adet, dlen, ddet, addet);
    return d_fast_expansion_sum_sign(bclen, bcdet, adlen, addet);
}

#undef MUL
#undef Fast_Two_Sum
#undef Two_Sum
#undef Two_Diff
#undef Split
#undef Two_Product
#undef Two_Product_Presplit
#undef Two_One_Diff
#undef Two_Two_Diff

}  // namespace spatial_predicates
