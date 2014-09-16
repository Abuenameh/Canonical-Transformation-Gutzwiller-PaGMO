#include <vector>
#include <cmath>
#include <complex>

using namespace std;

#include "gutzwiller.hpp"

static inline double Sqr(double x) {
    return x * x;
}

double energyfunc(const vector<double>& x, vector<double>& grad, void *data) {
    funcdata2* fdata = static_cast<funcdata2*> (data);
    vector<double>& U = fdata->U;
    vector<double>& J = fdata->J;
    double mu = fdata->mu;
    double theta = fdata->theta;

    doublecomplex expth = exp(doublecomplex(0, 1) * theta);
    doublecomplex expmth = exp(-doublecomplex(0, 1) * theta);
    doublecomplex exp2th = exp(doublecomplex(0, 1)*2.0 * theta);
    doublecomplex expm2th = exp(-doublecomplex(0, 1)*2.0 * theta);

    doublecomplex Ec = 0;

    const doublecomplex * f[L];
    vector<double> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += norm(f[i][n]);
        }
    }

    vector<doublecomplex> E0s(L, 0), E1j1s(L, 0), E1j2s(L, 0),
            E2j1s(L, 0), E2j2s(L, 0), E3j1s(L, 0), E3j2s(L, 0),
            E4j1j2s(L, 0), E4j1k1s(L, 0), E4j2k2s(L, 0),
            E5j1j2s(L, 0), E5j1k1s(L, 0), E5j2k2s(L, 0);


    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        doublecomplex E0 = 0;
        doublecomplex E1j1 = 0;
        doublecomplex E1j2 = 0;
        doublecomplex E2j1 = 0;
        doublecomplex E2j2 = 0;
        doublecomplex E3j1 = 0;
        doublecomplex E3j2 = 0;
        doublecomplex E4j1j2 = 0;
        doublecomplex E4j1k1 = 0;
        doublecomplex E4j2k2 = 0;
        doublecomplex E5j1j2 = 0;
        doublecomplex E5j1k1 = 0;
        doublecomplex E5j2k2 = 0;

        for (int n = 0; n <= nmax; n++) {
            E0 += (0.5 * U[i] * n * (n - 1) - mu * n + nu[i] * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                E1j1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                E1j2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    E2j1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1]
                            * (1 / eps(U, i, j1, n, n) - 1 / eps(U, i, j1, n - 1, n + 1));
                    E2j2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
                            * (1 / eps(U, i, j2, n, n) - 1 / eps(U, i, j2, n - 1, n + 1));
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        E3j1 += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]
                                - ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        E3j2 += 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]
                                - ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
                    }
                }

                if (n > 0) {
                    E4j1j2 += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
                            * f[i][n - 1] * f[j1][n] * f[j2][n + 1];
                    E4j1j2 += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
                            * f[i][n - 1] * f[j2][n] * f[j1][n + 1];
                    E4j1k1 += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][n + 1] * f[k1][n - 1];
                    E4j2k2 += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][n + 1] * f[k2][n - 1];
                    E4j1j2 -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                            * ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
                    E4j1j2 -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
                            * ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
                    E4j1k1 -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
                            * ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
                    E4j2k2 -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
                            * ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1 && n < nmax) {
                        E5j1j2 += 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
                        E5j1j2 += 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
                        E5j1k1 += 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
                        E5j2k2 += 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
                        E5j1j2 -= 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n] * f[j1][m] * f[j2][m - 1];
                        E5j1j2 -= 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n] * f[j2][m] * f[j1][m - 1];
                        E5j1k1 -= 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m]
                                * ~f[k1][n] * f[i][n] * f[j1][m] * f[k1][n + 1];
                        E5j2k2 -= 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m]
                                * ~f[k2][n] * f[i][n] * f[j2][m] * f[k2][n + 1];
                    }
                }
            }

        }

        Ec += E0 / norm2[i];
        
                Ec += E1j1 / (norm2[i] * norm2[j1]);
                Ec += E1j2 / (norm2[i] * norm2[j2]);
                
                Ec += E2j1 / (norm2[i] * norm2[j1]);
                Ec += E2j2 / (norm2[i] * norm2[j2]);
                
                Ec += E3j1 / (norm2[i] * norm2[j1]);
                Ec += E3j2 / (norm2[i] * norm2[j2]);
                
                Ec += E4j1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
                Ec += E4j1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
                Ec += E4j2k2 / (norm2[i] * norm2[j2] * norm2[k2]);
                
                Ec += E5j1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
                Ec += E5j1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
                Ec += E5j2k2 / (norm2[i] * norm2[j2] * norm2[k2]);

        E0s[i] += E0;
        
        E1j1s[i] += E1j1;
        E1j2s[i] += E1j2;
        
        E2j1s[i] += E2j1;
        E2j2s[i] += E2j2;
        
        E3j1s[i] += E3j1;
        E3j2s[i] += E3j2;
        
        E4j1j2s[i] += E4j1j2;
        E4j1k1s[i] += E4j1k1;
        E4j2k2s[i] += E4j2k2;
        
        E5j1j2s[i] += E5j1j2;
        E5j1k1s[i] += E5j1k1;
        E5j2k2s[i] += E5j2k2;
    }
    if (!grad.empty()) {
        for (int i = 0; i < L; i++) {

            int k1 = mod(i - 2);
            int j1 = mod(i - 1);
            int j2 = mod(i + 1);
            int k2 = mod(i + 2);
            for (int n = 0; n <= nmax; n++) {
                doublecomplex E0df = 0;
                doublecomplex E1j1df = 0;
                doublecomplex E1j2df = 0;
                doublecomplex E2j1df = 0;
                doublecomplex E2j2df = 0;
                doublecomplex E3j1df = 0;
                doublecomplex E3j2df = 0;
                doublecomplex E4j1j2df = 0;
                doublecomplex E4j1k1df = 0;
                doublecomplex E4j2k2df = 0;
                doublecomplex E5j1j2df = 0;
                doublecomplex E5j1k1df = 0;
                doublecomplex E5j2k2df = 0;

                E0df += (0.5 * U[i] * n * (n - 1) - mu * n + nu[i] * n) * f[i][n];

                if (n < nmax) {
                    E1j1df += -J[j1] * expmth * g(n, n + 1) * ~f[j1][n + 1] * f[j1][n]
                            * f[i][n + 1];
                    E1j2df += -J[i] * expth * g(n, n + 1) * ~f[j2][n + 1] * f[j2][n]
                            * f[i][n + 1];
                }
                if (n > 0) {
                    E1j1df += -J[j1] * expth * g(n - 1, n) * ~f[j1][n - 1] * f[j1][n]
                            * f[i][n - 1];
                    E1j2df += -J[i] * expmth * g(n - 1, n) * ~f[j2][n - 1] * f[j2][n]
                            * f[i][n - 1];
                }

                if (n > 1) {
                    E2j1df += 0.5 * J[j1] * J[j1] * exp2th * g(n - 1, n - 1) * g(n - 2, n)
                            * ~f[j1][n - 2] * f[j1][n] * f[i][n - 2]
                            * (1 / eps(U, i, j1, n - 1, n - 1) - 1 / eps(U, i, j1, n - 2, n));
                    E2j2df += 0.5 * J[i] * J[i] * expm2th * g(n - 1, n - 1) * g(n - 2, n)
                            * ~f[j2][n - 2] * f[j2][n] * f[i][n - 2]
                            * (1 / eps(U, i, j2, n - 1, n - 1) - 1 / eps(U, i, j2, n - 2, n));
                }
                if (n < nmax - 1) {
                    E2j1df += 0.5 * J[j1] * J[j1] * expm2th * g(n + 1, n + 1) * g(n, n + 2)
                            * ~f[j1][n + 2] * f[j1][n] * f[i][n + 2]
                            * (1 / eps(U, j1, i, n + 1, n + 1) - 1 / eps(U, j1, i, n, n + 2));
                    E2j2df += 0.5 * J[i] * J[i] * exp2th * g(n + 1, n + 1) * g(n, n + 2)
                            * ~f[j2][n + 2] * f[j2][n] * f[i][n + 2]
                            * (1 / eps(U, j2, i, n + 1, n + 1) - 1 / eps(U, j2, i, n, n + 2));
                }

                for (int m = 0; m < nmax; m++) {
                    if (n != m + 1) {
                        E3j1df += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n - 1, m + 1))
                                * g(n - 1, m + 1) * g(m, n) * ~f[j1][m] * f[j1][m] * f[i][n];
                        E3j2df += 0.5 * (J[i] * J[i] / eps(U, i, j2, n - 1, m + 1))
                                * g(n - 1, m + 1) * g(m, n) * ~f[j2][m] * f[j2][m] * f[i][n];
                        E3j1df -= 0.5 * (J[j1] * J[j1] / eps(U, j1, i, m, n)) * g(m, n)
                                * g(n - 1, m + 1) * ~f[j1][m] * f[j1][m] * f[i][n];
                        E3j2df -= 0.5 * (J[i] * J[i] / eps(U, j2, i, m, n)) * g(m, n)
                                * g(n - 1, m + 1) * ~f[j2][m] * f[j2][m] * f[i][n];
                    }
                    if (n != m && n < nmax) {
                        E3j1df += 0.5 * (J[j1] * J[j1] / eps(U, j1, i, m, n + 1))
                                * g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1] * f[j1][m + 1]
                                * f[i][n];
                        E3j2df += 0.5 * (J[i] * J[i] / eps(U, j2, i, m, n + 1))
                                * g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1] * f[j2][m + 1]
                                * f[i][n];
                        E3j1df -= 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m + 1))
                                * g(n, m + 1) * g(m, n + 1) * ~f[j1][m + 1] * f[j1][m + 1]
                                * f[i][n];
                        E3j2df -= 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m + 1))
                                * g(n, m + 1) * g(m, n + 1) * ~f[j2][m + 1] * f[j2][m + 1]
                                * f[i][n];
                    }
                }

                if (n >= 2) {
                    E4j1j2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[j1] * J[i] / eps(U, i, j1, n - 1, n - 1)) * ~f[j1][n - 2]
                            * ~f[j2][n - 1] * f[j1][n - 1] * f[j2][n] * f[i][n - 2];
                    E4j1j2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[i] * J[j1] / eps(U, i, j2, n - 1, n - 1)) * ~f[j2][n - 2]
                            * ~f[j1][n - 1] * f[j2][n - 1] * f[j1][n] * f[i][n - 2];
                    E4j1k1df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n - 1)) * ~f[j1][n - 2]
                            * ~f[k1][n - 1] * f[j1][n] * f[k1][n - 2] * f[i][n - 1];
                    E4j2k2df += 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[i] * J[j2] / eps(U, i, j2, n - 1, n - 1)) * ~f[j2][n - 2]
                            * ~f[k2][n - 1] * f[j2][n] * f[k2][n - 2] * f[i][n - 1];
                    E4j1j2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[j1] * J[i] / eps(U, i, j1, n - 2, n)) * ~f[j1][n - 1]
                            * ~f[j2][n - 2] * f[j1][n] * f[j2][n - 1] * f[i][n - 2];
                    E4j1j2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[i] * J[j1] / eps(U, i, j2, n - 2, n)) * ~f[j2][n - 1]
                            * ~f[j1][n - 2] * f[j2][n] * f[j1][n - 1] * f[i][n - 2];
                    E4j1k1df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[k1] * J[j1] / eps(U, k1, j1, n - 2, n)) * ~f[k1][n - 1]
                            * ~f[j1][n - 2] * f[k1][n - 2] * f[j1][n] * f[i][n - 1];
                    E4j2k2df -= 0.5 * g(n - 1, n - 1) * g(n - 2, n)
                            * (J[j2] * J[i] / eps(U, k2, j2, n - 2, n)) * ~f[k2][n - 1]
                            * ~f[j2][n - 2] * f[k2][n - 2] * f[j2][n] * f[i][n - 1];
                }
                if (n >= 1 && n <= nmax - 1) {
                    E4j1k1df += 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[k1] * J[j1] / eps(U, j1, k1, n, n)) * ~f[j1][n + 1]
                            * ~f[k1][n - 1] * f[j1][n - 1] * f[k1][n] * f[i][n + 1];
                    E4j2k2df += 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[j2] * J[i] / eps(U, j2, k2, n, n)) * ~f[j2][n + 1]
                            * ~f[k2][n - 1] * f[j2][n - 1] * f[k2][n] * f[i][n + 1];
                    E4j1k1df += 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[k1] * J[j1] / eps(U, k1, j1, n, n)) * ~f[k1][n + 1]
                            * ~f[j1][n - 1] * f[k1][n] * f[j1][n + 1] * f[i][n - 1];
                    E4j2k2df += 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[j2] * J[i] / eps(U, k2, j2, n, n)) * ~f[k2][n + 1]
                            * ~f[j2][n - 1] * f[k2][n] * f[j2][n + 1] * f[i][n - 1];
                    E4j1k1df -= 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[j1] * J[k1] / eps(U, j1, i, n - 1, n + 1)) * ~f[j1][n + 1]
                            * ~f[k1][n - 1] * f[j1][n - 1] * f[k1][n] * f[i][n + 1];
                    E4j2k2df -= 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[i] * J[j2] / eps(U, j2, i, n - 1, n + 1)) * ~f[j2][n + 1]
                            * ~f[k2][n - 1] * f[j2][n - 1] * f[k2][n] * f[i][n + 1];
                    E4j1k1df -= 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1)) * ~f[j1][n - 1]
                            * ~f[k1][n + 1] * f[j1][n + 1] * f[k1][n] * f[i][n - 1];
                    E4j2k2df -= 0.5 * g(n, n) * g(n - 1, n + 1)
                            * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1)) * ~f[j2][n - 1]
                            * ~f[k2][n + 1] * f[j2][n + 1] * f[k2][n] * f[i][n - 1];
                }
                if (n <= nmax - 2) {
                    E4j1k1df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[j1] * J[k1] / eps(U, j1, i, n + 1, n + 1)) * ~f[j1][n + 2]
                            * ~f[k1][n + 1] * f[j1][n] * f[k1][n + 2] * f[i][n + 1];
                    E4j2k2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[i] * J[j2] / eps(U, j2, i, n + 1, n + 1)) * ~f[j2][n + 2]
                            * ~f[k2][n + 1] * f[j2][n] * f[k2][n + 2] * f[i][n + 1];
                    E4j1j2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[j1] * J[i] / eps(U, j1, i, n + 1, n + 1)) * ~f[j1][n + 2]
                            * ~f[j2][n + 1] * f[j1][n + 1] * f[j2][n] * f[i][n + 2];
                    E4j1j2df += 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[i] * J[j1] / eps(U, j2, i, n + 1, n + 1)) * ~f[j2][n + 2]
                            * ~f[j1][n + 1] * f[j2][n + 1] * f[j1][n] * f[i][n + 2];
                    E4j1k1df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[k1] * J[j1] / eps(U, j1, k1, n, n + 2)) * ~f[j1][n + 2]
                            * ~f[k1][n + 1] * f[j1][n] * f[k1][n + 2] * f[i][n + 1];
                    E4j2k2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[j2] * J[i] / eps(U, j2, k2, n, n + 2)) * ~f[j2][n + 2]
                            * ~f[k2][n + 1] * f[j2][n] * f[k2][n + 2] * f[i][n + 1];
                    E4j1j2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[j1] * J[i] / eps(U, j1, i, n, n + 2)) * ~f[j1][n + 1]
                            * ~f[j2][n + 2] * f[j1][n] * f[j2][n + 1] * f[i][n + 2];
                    E4j1j2df -= 0.5 * g(n + 1, n + 1) * g(n, n + 2)
                            * (J[i] * J[j1] / eps(U, j2, i, n, n + 2)) * ~f[j2][n + 1]
                            * ~f[j1][n + 2] * f[j2][n] * f[j1][n + 1] * f[i][n + 2];
                }

                for (int m = 0; m <= nmax; m++) {
                    if (n != m) {
                        if (n > 0) {
                            if (m > 0) {
                                E5j1j2df += 0.5
                                        * (J[j1] * J[i] * exp2th / eps(U, i, j1, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1] * ~f[j2][m]
                                        * f[j1][m] * f[j2][m - 1] * f[i][n];
                                E5j1j2df += 0.5
                                        * (J[i] * J[j1] * expm2th / eps(U, i, j2, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1] * ~f[j1][m]
                                        * f[j2][m] * f[j1][m - 1] * f[i][n];
                                E5j1k1df += 0.5
                                        * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j1][m - 1]
                                        * ~f[k1][n - 1] * f[j1][m - 1] * f[k1][n] * f[i][n - 1];
                                E5j2k2df += 0.5
                                        * (J[i] * J[j2] * expm2th / eps(U, i, j2, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j2][m - 1]
                                        * ~f[k2][n - 1] * f[j2][m - 1] * f[k2][n] * f[i][n - 1];
                                E5j1k1df -= 0.5
                                        * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j1][m] * ~f[k1][n - 1]
                                        * f[j1][m] * f[k1][n] * f[i][n - 1];
                                E5j2k2df -= 0.5
                                        * (J[i] * J[j2] * expm2th / eps(U, i, j2, n - 1, m))
                                        * g(n - 1, m) * g(m - 1, n) * ~f[j2][m] * ~f[k2][n - 1]
                                        * f[j2][m] * f[k2][n] * f[i][n - 1];
                            }
                        }
                        if (n < nmax) {
                            if (m < nmax) {
                                E5j1k1df += 0.5
                                        * (J[j1] * J[k1] * expm2th / eps(U, j1, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1]
                                        * ~f[k1][n + 1] * f[j1][m + 1] * f[k1][n] * f[i][n + 1];
                                E5j2k2df += 0.5
                                        * (J[i] * J[j2] * exp2th / eps(U, j2, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1]
                                        * ~f[k2][n + 1] * f[j2][m + 1] * f[k2][n] * f[i][n + 1];
                                E5j1j2df += 0.5
                                        * (J[j1] * J[i] * expm2th / eps(U, j1, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
                                        * f[j1][m] * f[j2][m + 1] * f[i][n];
                                E5j1j2df += 0.5
                                        * (J[i] * J[j1] * exp2th / eps(U, j2, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
                                        * f[j2][m] * f[j1][m + 1] * f[i][n];
                                E5j1k1df -= 0.5
                                        * (J[j1] * J[k1] * expm2th / eps(U, j1, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j1][m] * ~f[k1][n + 1]
                                        * f[j1][m] * f[k1][n] * f[i][n + 1];
                                E5j2k2df -= 0.5
                                        * (J[i] * J[j2] * exp2th / eps(U, j2, i, m, n + 1))
                                        * g(m, n + 1) * g(n, m + 1) * ~f[j2][m] * ~f[k2][n + 1]
                                        * f[j2][m] * f[k2][n] * f[i][n + 1];
                            }
                        }
                    }
                    if (n != m + 1) {
                        if (m < nmax) {
                            E5j1j2df -= 0.5 * (J[j1] * J[i] * expm2th / eps(U, j1, i, m, n))
                                    * g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1] * ~f[j2][m]
                                    * f[j1][m] * f[j2][m + 1] * f[i][n];
                            E5j1j2df -= 0.5 * (J[i] * J[j1] * exp2th / eps(U, j2, i, m, n))
                                    * g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1] * ~f[j1][m]
                                    * f[j2][m] * f[j1][m + 1] * f[i][n];
                        }
                        if (n > 0) {
                            if (m < nmax) {
                                E5j1k1df += 0.5
                                        * (J[k1] * J[j1] * exp2th / eps(U, j1, k1, m, n))
                                        * g(m, n) * g(n - 1, m + 1) * ~f[j1][m + 1]
                                        * ~f[k1][n - 1] * f[j1][m + 1] * f[k1][n] * f[i][n - 1];
                                E5j2k2df += 0.5
                                        * (J[j2] * J[i] * expm2th / eps(U, j2, k2, m, n))
                                        * g(m, n) * g(n - 1, m + 1) * ~f[j2][m + 1]
                                        * ~f[k2][n - 1] * f[j2][m + 1] * f[k2][n] * f[i][n - 1];
                                E5j1k1df -= 0.5
                                        * (J[k1] * J[j1] * exp2th / eps(U, j1, k1, m, n))
                                        * g(m, n) * g(n - 1, m + 1) * ~f[j1][m] * ~f[k1][n - 1]
                                        * f[j1][m] * f[k1][n] * f[i][n - 1];
                                E5j2k2df -= 0.5
                                        * (J[j2] * J[i] * expm2th / eps(U, j2, k2, m, n))
                                        * g(m, n) * g(n - 1, m + 1) * ~f[j2][m] * ~f[k2][n - 1]
                                        * f[j2][m] * f[k2][n] * f[i][n - 1];
                            }
                        }
                    }
                    if (n != m - 1) {
                        if (n < nmax) {
                            if (m > 0) {
                                E5j1k1df += 0.5
                                        * (J[k1] * J[j1] * expm2th / eps(U, k1, j1, n, m))
                                        * g(n, m) * g(m - 1, n + 1) * ~f[k1][n + 1]
                                        * ~f[j1][m - 1] * f[k1][n] * f[j1][m - 1] * f[i][n + 1];
                                E5j2k2df += 0.5
                                        * (J[j2] * J[i] * exp2th / eps(U, k2, j2, n, m))
                                        * g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1]
                                        * ~f[j2][m - 1] * f[k2][n] * f[j2][m - 1] * f[i][n + 1];
                                E5j1j2df -= 0.5
                                        * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                                        * g(n, m) * g(m - 1, n + 1) * ~f[j1][m - 1] * ~f[j2][m]
                                        * f[j1][m] * f[j2][m - 1] * f[i][n];
                                E5j1j2df -= 0.5
                                        * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                                        * g(n, m) * g(m - 1, n + 1) * ~f[j2][m - 1] * ~f[j1][m]
                                        * f[j2][m] * f[j1][m - 1] * f[i][n];
                            }
                            E5j1k1df -= 0.5
                                    * (J[k1] * J[j1] * expm2th / eps(U, k1, j1, n, m)) * g(n, m)
                                    * g(m - 1, n + 1) * ~f[k1][n + 1] * ~f[j1][m] * f[k1][n]
                                    * f[j1][m] * f[i][n + 1];
                            E5j2k2df -= 0.5 * (J[j2] * J[i] * exp2th / eps(U, k2, j2, n, m))
                                    * g(n, m) * g(m - 1, n + 1) * ~f[k2][n + 1] * ~f[j2][m]
                                    * f[k2][n] * f[j2][m] * f[i][n + 1];
                        }
                    }
                }

                doublecomplex Edf = 0;

                Edf += (E0df * norm2[i] - E0s[i] * f[i][n]) / (norm2[i] * norm2[i]);

                                Edf += (E1j1df * norm2[i] * norm2[j1]
                                        - (E1j1s[i] + E1j2s[j1]) * f[i][n] * norm2[j1])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
                                Edf += (E1j2df * norm2[i] * norm2[j2]
                                        - (E1j2s[i] + E1j1s[j2]) * f[i][n] * norm2[j2])
                                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);
                
                                Edf += (E2j1df * norm2[i] * norm2[j1]
                                        - (E2j1s[i] + E2j2s[j1]) * f[i][n] * norm2[j1])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
                                Edf += (E2j2df * norm2[i] * norm2[j2]
                                        - (E2j2s[i] + E2j1s[j2]) * f[i][n] * norm2[j2])
                                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);
                
                                Edf += (E3j1df * norm2[i] * norm2[j1]
                                        - (E3j1s[i] + E3j2s[j1]) * f[i][n] * norm2[j1])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
                                Edf += (E3j2df * norm2[i] * norm2[j2]
                                        - (E3j2s[i] + E3j1s[j2]) * f[i][n] * norm2[j2])
                                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);
                
                                Edf += (E4j1j2df * norm2[i] * norm2[j1] * norm2[j2]
                                        - (E4j1j2s[i] + E4j2k2s[j1] + E4j1k1s[j2]) * f[i][n] * norm2[j1]
                                        * norm2[j2])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[j2] * norm2[j2]);
                                Edf += (E4j1k1df * norm2[i] * norm2[j1] * norm2[k1]
                                        - (E4j1k1s[i] + E4j1j2s[j1] + E4j2k2s[k1]) * f[i][n] * norm2[j1]
                                        * norm2[k1])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[k1] * norm2[k1]);
                                Edf += (E4j2k2df * norm2[i] * norm2[j2] * norm2[k2]
                                        - (E4j2k2s[i] + E4j1j2s[j2] + E4j1k1s[k2]) * f[i][n] * norm2[j2]
                                        * norm2[k2])
                                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2] * norm2[k2] * norm2[k2]);
                
                                Edf += (E5j1j2df * norm2[i] * norm2[j1] * norm2[j2]
                                        - (E5j1j2s[i] + E5j2k2s[j1] + E5j1k1s[j2]) * f[i][n] * norm2[j1]
                                        * norm2[j2])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[j2] * norm2[j2]);
                                Edf += (E5j1k1df * norm2[i] * norm2[j1] * norm2[k1]
                                        - (E5j1k1s[i] + E5j1j2s[j1] + E5j2k2s[k1]) * f[i][n] * norm2[j1]
                                        * norm2[k1])
                                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1] * norm2[k1] * norm2[k1]);
                                Edf += (E5j2k2df * norm2[i] * norm2[j2] * norm2[k2]
                                        - (E5j2k2s[i] + E5j1j2s[j2] + E5j1k1s[k2]) * f[i][n] * norm2[j2]
                                        * norm2[k2])
                                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2] * norm2[k2] * norm2[k2]);

                int k = i * dim + n;
                grad[2 * k] = 2 * Edf.real();
                grad[2 * k + 1] = 2 * Edf.imag();
                
            }
        }
    }
        return Ec.real();
}
