#pragma once
#include <cuda_runtime.h>

namespace spatial_predicates {

using RealType = double;
constexpr int Splitter = 0;
__device__ __constant__ double constants[] = {134217729.0};

#define MUL(a, b) __dmul_rn(a, b)

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a;                    \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (RealType)(a + b);       \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (RealType)(x - a);   \
    avirt = x - bvirt;           \
    bround = b - bvirt;          \
    around = a - avirt;          \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (RealType)(a + b);  \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (RealType)(a - x);    \
    avirt = x + bvirt;            \
    bround = bvirt - b;           \
    around = a - avirt;           \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (RealType)(a - b);   \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo)            \
    c = MUL(predConsts[Splitter], a); \
    abig = (RealType)(c - a);         \
    ahi = c - abig;                   \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo);              \
    Split(b, bhi, blo);              \
    err1 = x - MUL(ahi, bhi);        \
    err2 = err1 - MUL(alo, bhi);     \
    err3 = err2 - MUL(ahi, blo);     \
    y = MUL(alo, blo) - err3

#define Two_Product(a, b, x, y) \
    x = MUL(a, b);              \
    Two_Product_Tail(a, b, x, y)

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

__device__ int d_fast_expansion_sum_zeroelim(int elen, RealType* e, int flen,
                                             RealType* f, RealType* h) {
    RealType Q;
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex, hindex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        if (++eindex < elen) enow = e[eindex];
    } else {
        Q = fnow;
        if (++findex < flen) fnow = f[findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            if (++eindex < elen) enow = e[eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            if (++findex < flen) fnow = f[findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                if (++eindex < elen) enow = e[eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                if (++findex < flen) fnow = f[findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

__device__ RealType d_fast_expansion_sum_sign(int elen, RealType* e, int flen,
                                              RealType* f) {
    RealType Q;
    RealType lastTerm;
    RealType Qnew;
    RealType hh;
    RealType bvirt;
    RealType avirt, bround, around;
    int eindex, findex;
    RealType enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        if (++eindex < elen) enow = e[eindex];
    } else {
        Q = fnow;
        if (++findex < flen) fnow = f[findex];
    }
    lastTerm = 0.0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            if (++eindex < elen) enow = e[eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            if (++findex < flen) fnow = f[findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                if (++eindex < elen) enow = e[eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                if (++findex < flen) fnow = f[findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                lastTerm = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        if (++eindex < elen) enow = e[eindex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        if (++findex < flen) fnow = f[findex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    if (Q != 0.0) {
        lastTerm = Q;
    }
    return lastTerm;
}

__device__ int d_scale_twice_expansion_zeroelim(const RealType* predConsts,
                                                int elen, RealType* e,
                                                RealType b1, RealType b2,
                                                RealType* h) {
    RealType Q, sum, Q2, sum2;
    RealType hh;
    RealType product1, product2;
    RealType product0;
    int eindex, hindex;
    RealType enow;
    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, b1hi, b1lo, b2hi, b2lo;
    RealType err1, err2, err3;

    hindex = 0;

    Split(b1, b1hi, b1lo);
    Split(b2, b2hi, b2lo);
    Two_Product_Presplit(e[0], b1, b1hi, b1lo, Q, hh);
    Two_Product_Presplit(hh, b2, b2hi, b2lo, Q2, hh);

    if (hh != 0) {
        h[hindex++] = hh;
    }

    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b1, b1hi, b1lo, product1, product0);
        Two_Sum(Q, product0, sum, hh);

        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product1, sum, Q, hh);

        Two_Product_Presplit(hh, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if (Q != 0) {
        Two_Product_Presplit(Q, b2, b2hi, b2lo, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);

        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if ((Q2 != 0) || (hindex == 0)) {
        h[hindex++] = Q2;
    }

    return hindex;
}

__noinline__ __device__ RealType orient2dExact(const RealType* predConsts,
                                               const RealType* pa,
                                               const RealType* pb,
                                               const RealType* pc) {
    RealType axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
    RealType axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
    RealType aterms[4], bterms[4], cterms[4];
    RealType v[8];
    int vlength;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

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

    vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);

    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

__forceinline__ __device__ void two_mult_sub(const RealType* predConsts,
                                             const RealType* pa,
                                             const RealType* pb, RealType* ab) {
    RealType axby1, axby0, bxay1, bxay0;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);
}

__noinline__ __device__ int calc_det(const RealType* predConsts, RealType* a,
                                     RealType* b, RealType* c, RealType fx0,
                                     RealType fx1, RealType fy0, RealType fy1,
                                     RealType* temp2, RealType* temp3,
                                     RealType* detx, RealType* dety,
                                     RealType* ret) {
    int temp2len = d_fast_expansion_sum_zeroelim(4, a, 4, b, temp2);
    int temp3len = d_fast_expansion_sum_zeroelim(temp2len, temp2, 4, c, temp3);

    int xlen = d_scale_twice_expansion_zeroelim(predConsts, temp3len, temp3,
                                                fx0, fx1, detx);
    int ylen = d_scale_twice_expansion_zeroelim(predConsts, temp3len, temp3,
                                                fy0, fy1, dety);

    return d_fast_expansion_sum_zeroelim(xlen, detx, ylen, dety, ret);
}

__noinline__ __device__ RealType incircleExact(const RealType* predConsts,
                                               const RealType* pa,
                                               const RealType* pb,
                                               const RealType* pc,
                                               const RealType* pd) {
    RealType ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    RealType temp8[8];
    RealType temp12[12];
    RealType det48x[48], det48y[48];
    RealType bdet[96], cdet[96];
    int alen, blen, clen, dlen;
    RealType bcdet[192], addet[192];
    int bclen, adlen;
    int i;

    RealType* adet = bdet;
    RealType* ddet = cdet;

    RealType bvirt;
    RealType avirt, bround, around;
    RealType c;
    RealType abig;
    RealType ahi, alo, bhi, blo;
    RealType err1, err2, err3;
    RealType _i, _j;
    RealType _0;

    two_mult_sub(predConsts, pa, pb, ab);
    two_mult_sub(predConsts, pb, pc, bc);
    two_mult_sub(predConsts, pc, pd, cd);
    two_mult_sub(predConsts, pd, pa, da);
    two_mult_sub(predConsts, pa, pc, ac);
    two_mult_sub(predConsts, pb, pd, bd);

    blen = calc_det(predConsts, cd, da, ac, pb[0], -pb[0], pb[1], -pb[1], temp8,
                    temp12, det48x, det48y, bdet);
    clen = calc_det(predConsts, da, ab, bd, pc[0], pc[0], pc[1], pc[1], temp8,
                    temp12, det48x, det48y, cdet);

    bclen = d_fast_expansion_sum_zeroelim(blen, bdet, clen, cdet, bcdet);

    for (i = 0; i < 4; i++) {
        bd[i] = -bd[i];
        ac[i] = -ac[i];
    }

    dlen = calc_det(predConsts, ab, bc, ac, pd[0], -pd[0], pd[1], -pd[1], temp8,
                    temp12, det48x, det48y, ddet);
    alen = calc_det(predConsts, bc, cd, bd, pa[0], pa[0], pa[1], pa[1], temp8,
                    temp12, det48x, det48y, adet);

    adlen = d_fast_expansion_sum_zeroelim(alen, adet, dlen, ddet, addet);

    return d_fast_expansion_sum_sign(bclen, bcdet, adlen, addet);
}

#undef MUL
#undef Fast_Two_Sum_Tail
#undef Fast_Two_Sum
#undef Two_Sum_Tail
#undef Two_Sum
#undef Two_Diff_Tail
#undef Two_Diff
#undef Split
#undef Two_Product_Tail
#undef Two_Product
#undef Two_Product_Presplit
#undef Two_One_Diff
#undef Two_Two_Diff

}  // namespace spatial_predicates
