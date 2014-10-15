#include <pagmo/src/pagmo.h>

#include "gutzwiller.hpp"

using namespace std;
using namespace pagmo;
using namespace pagmo::problem;

//static inline double Sqr(double x) {
//    return x * x;
//}
//
//class energy : public base {
//public:
//    energy(int n, vector<double>& U_, vector<double>& J_, double mu_, double theta_) : base(n), U(U_), J(J_), mu(mu_), theta(theta_) {
//        costh = cos(theta);
//        sinth = sin(theta);
//        cos2th = cos(2*theta);
//        sin2th = sin(2*theta);
//    }
//    
//    base_ptr clone() const;
//    void objfun_impl(fitness_vector& f, const decision_vector& x) const;
//    
//private:
//    vector<double>& U;
//    vector<double>& J;
//    double mu;
//    double theta;
//    double costh;
//    double sinth;
//    double cos2th;
//    double sin2th;
//};

base_ptr energyprob::clone() const {
    return base_ptr(new energyprob(*this));
}

#define NEW

void energyprob::objfun_impl(fitness_vector& En, const decision_vector& x) const {
    En[0] = energy(const_cast<double*>(x.data()), U.data(), U0, dU.data(), J.data(), mu, theta);
}

void energyprob::objfun_impl4(fitness_vector& En, const decision_vector& x) const {
#ifdef NEW
    objfun_implnew(En, x);
#else
    objfun_implold(En, x);
#endif
}

void energyprob::objfun_implnew(fitness_vector& En, const decision_vector& x) const {
    En[0] = energy(const_cast<double*>(x.data()), U.data(), U0, dU.data(), J.data(), mu, theta);
//    cout << En[0] << endl;
}

void energyprob::objfun_implold(fitness_vector& En, const decision_vector& x) const {

    doublecomplex expth = exp(doublecomplex(0, 1) * theta);
    doublecomplex expmth = ~expth;
    doublecomplex exp2th = expth*expth;
    doublecomplex expm2th = ~exp2th;

    doublecomplex Ec = 0;

    const doublecomplex * f[L];
    vector<double> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += norm(f[i][n]);
        }
    }

    doublecomplex qwe = 0;

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
        doublecomplex E5j1k1 = 0;
        doublecomplex E5j2k2 = 0;

        doublecomplex dE0_1j1 = 0;
        doublecomplex dE0_1j2 = 0;

        doublecomplex dE1_2j1 = 0;
        doublecomplex dE1_2j2 = 0;
        doublecomplex dE1_3j1 = 0;
        doublecomplex dE1_3j2 = 0;
        doublecomplex dE1_4j1j2 = 0;
        doublecomplex dE1_5j1k1 = 0;
        doublecomplex dE1_5j2k2 = 0;

        doublecomplex dE2_2j1 = 0;
        doublecomplex dE2_2j2 = 0;
        doublecomplex dE2_4j1j2 = 0;
        doublecomplex dE2_5j1k1 = 0;
        doublecomplex dE2_5j2k2 = 0;

        doublecomplex dE3_2j1 = 0;
        doublecomplex dE3_2j2 = 0;
        doublecomplex dE3_4j1j2 = 0;
        doublecomplex dE3_5j1k1 = 0;
        doublecomplex dE3_5j2k2 = 0;

        for (int n = 0; n <= nmax; n++) {
            E0 += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                E1j1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                E1j2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    E2j1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                    E2j2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
                }
                if (n < nmax - 1) {
                    E2j1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                    E2j2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
                }

                if (n > 1) {
                    dE2_2j1 += -J[j1] * J[j1] * exp2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                    dE2_2j2 += -J[i] * J[i] * expm2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
                }
                if (n < nmax - 2) {
                    dE2_2j1 -= -J[j1] * J[j1] * exp2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                    dE2_2j2 -= -J[i] * J[i] * expm2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j2, n, n + 3, i, j2, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j2][n + 1] * f[i][n] * f[j2][n + 3];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        E3j1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                                ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        E3j2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                                ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                        dE0_1j1 += J[j1] * expth * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        dE0_1j2 += J[i] * expmth * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                        if (n != m - 3 && m > 1 && n < nmax - 1) {
                            dE1_2j1 += -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            dE1_2j2 += -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                        if (n != m + 1 && n > 0 && m < nmax) {
                            dE1_2j1 -= -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            dE1_2j2 -= -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }

                        if (n > 0) {
                            dE2_4j1j2 += -J[j1] * J[i] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j1, n, m, i, j2, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][n - 1]
                                    * f[i][n - 1] * f[j1][m] * f[j2][n];
                            dE2_4j1j2 += -J[i] * J[j1] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j2, n, m, i, j1, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][n - 1]
                                    * f[i][n - 1] * f[j2][m] * f[j1][n];
                        }
                        if (n < nmax - 1) {
                            dE2_4j1j2 -= -J[j1] * J[i] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j1, n, m, i, j2, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][n + 1]
                                    * f[i][n] * f[j1][m] * f[j2][n + 2];
                            dE2_4j1j2 -= -J[i] * J[j1] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j2, n, m, i, j1, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][n + 1]
                                    * f[i][n] * f[j2][m] * f[j1][n + 2];
                        }

                        dE1_3j1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m] -
                                ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]);
                        dE1_3j2 += -0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m] -
                                ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]);

                        for (int q = 1; q <= nmax; q++) {
                            if (n < nmax - 1 && n != q - 2) {
                                dE1_4j1j2 += -0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n] * f[j1][m] * f[j2][q];
                                dE1_4j1j2 += -0.5 * J[i] * J[j1] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n] * f[j2][m] * f[j1][q];
                            }
                            if (n > 0 && n != q) {
                                dE1_4j1j2 -= -0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n - 1] * f[j1][m] * f[j2][q];
                                dE1_4j1j2 -= -0.5 * J[i] * J[j1] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n - 1] * f[j2][m] * f[j1][q];
                            }

                            if (m != q) {
                                dE1_5j1k1 += -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m] * f[k1][q];
                                dE1_5j2k2 += -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m] * f[k2][q];
                                dE1_5j1k1 -= -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m - 1] * f[k1][q];
                                dE1_5j2k2 -= -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m - 1] * f[k2][q];
                            }

                        }

                        for (int p = 0; p < nmax; p++) {

                            if (p != n - 1 && 2 * n - m == p && n > 0) {
                                E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                                E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != n - 1 && 2 * n - m != p && n > 0) {
                                dE3_4j1j2 += -0.25 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                dE3_4j1j2 += -0.25 * J[i] * J[j1] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m != p - 2 && n < nmax - 1) {
                                dE3_4j1j2 -= -0.25 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                dE3_4j1j2 -= -0.25 * J[i] * J[j1] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != m - 1 && n != p) {
                                dE3_5j1k1 += -0.25 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                        ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                                dE3_5j2k2 += -0.25 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                        ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                            }
                        }

                        E5j1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                                * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                                * f[i][n] * f[j1][m] * f[k1][n + 1]);
                        E5j2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                                * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                                * f[i][n] * f[j2][m] * f[k2][n + 1]);

                        dE2_5j1k1 += -J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                        dE2_5j2k2 += -J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                        if (m != n - 1 && n != m && m < nmax && n > 0) {
                            dE3_2j1 += -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            dE3_2j2 += -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }
                        if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                            dE3_2j1 -= -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            dE3_2j2 -= -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }


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

        Ec += (E4j1j2 + ~E4j1j2) / (norm2[i] * norm2[j1] * norm2[j2]);

        Ec += (E5j1k1 + ~E5j1k1) / (norm2[i] * norm2[j1] * norm2[k1]);
        Ec += (E5j2k2 + ~E5j2k2) / (norm2[i] * norm2[j2] * norm2[k2]);

        Ec += dE0_1j1 / (norm2[i] * norm2[j1]);
        Ec += dE0_1j2 / (norm2[i] * norm2[j2]);

        Ec += dE1_2j1 / (norm2[i] * norm2[j1]);
        Ec += dE1_2j2 / (norm2[i] * norm2[j2]);

        Ec += dE1_3j1 / (norm2[i] * norm2[j1]);
        Ec += dE1_3j2 / (norm2[i] * norm2[j2]);

        Ec += (dE1_4j1j2 + ~dE1_4j1j2) / (norm2[i] * norm2[j1] * norm2[j2]);

        Ec += (dE1_5j1k1 + ~dE1_5j1k1) / (norm2[i] * norm2[j1] * norm2[k1]);
        Ec += (dE1_5j2k2 + ~dE1_5j2k2) / (norm2[i] * norm2[j2] * norm2[k2]);

        Ec += dE2_2j1 / (norm2[i] * norm2[j1]);
        Ec += dE2_2j2 / (norm2[i] * norm2[j2]);

        Ec += (dE2_4j1j2 + ~dE2_4j1j2) / (norm2[i] * norm2[j1] * norm2[j2]);

        Ec += (dE2_5j1k1 + ~dE2_5j1k1) / (norm2[i] * norm2[j1] * norm2[k1]);
        Ec += (dE2_5j2k2 + ~dE2_5j2k2) / (norm2[i] * norm2[j2] * norm2[k2]);

        Ec += (dE3_2j1 + ~dE3_2j1) / (norm2[i] * norm2[j1]);
        Ec += (dE3_2j2 + ~dE3_2j2) / (norm2[i] * norm2[j2]);

        Ec += 2.0 * (dE3_4j1j2 + ~dE3_4j1j2) / (norm2[i] * norm2[j1] * norm2[j2]);

        Ec += 2.0 * (dE3_5j1k1 + ~dE3_5j1k1) / (norm2[i] * norm2[j1] * norm2[k1]);
        Ec += 2.0 * (dE3_5j2k2 + ~dE3_5j2k2) / (norm2[i] * norm2[j2] * norm2[k2]);
    }
    //    cout << qwe << endl;
    //    exit(0);

//    cout << Ec << endl;
    En[0] = Ec.real();
//    count[0]++;
//    if(count[0]%100==0) {
//        cout << count[0] << endl;
//    }
    //    static int count = 0;
    //    cout << count++ << endl;

    //#include "vars.cpp"
    //#include "return.cpp"
    //    
    //    En[0] = E;
}

void energyprob::objfun_impl2(fitness_vector& En, const decision_vector& x) const {

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
            E0 += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

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

                    E5j1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U, i, j1, n, m))
                            * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                            ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                            * f[i][n] * f[j1][m] * f[k1][n + 1]);
                    E5j2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U, i, j2, n, m))
                            * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                            ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                            * f[i][n] * f[j2][m] * f[k2][n + 1]);
                    }

                    for (int p = 0; p < nmax; p++) {

                        if (p != n - 1 && 2 * n - m == p && n > 0) {
                            E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U, i, j1, n, m))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                            E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U, i, j2, n, m))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                        }
                        if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                            E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U, i, j1, n, m))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                    * f[i][n] * f[j1][m] * f[j2][p + 1];
                            E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U, i, j2, n, m))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                    * f[i][n] * f[j2][m] * f[j1][p + 1];
                        }
                    }
                }


                //                if (n > 0) {
                //                    E4j1j2 += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
                //                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
                //                            * f[i][n - 1] * f[j1][n] * f[j2][n + 1];
                //                    E4j1j2 += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
                //                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
                //                            * f[i][n - 1] * f[j2][n] * f[j1][n + 1];
                //                    E4j1k1 += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
                //                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
                //                            * f[i][n] * f[j1][n + 1] * f[k1][n - 1];
                //                    E4j2k2 += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
                //                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
                //                            * f[i][n] * f[j2][n + 1] * f[k2][n - 1];
                //                    E4j1j2 -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1))
                //                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                //                            * ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
                //                    E4j1j2 -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1))
                //                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
                //                            * ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
                //                    E4j1k1 -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1))
                //                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
                //                            * ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
                //                    E4j2k2 -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1))
                //                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
                //                            * ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
                //                }

                //                for (int m = 1; m <= nmax; m++) {
                //                    if (n != m - 1 && n < nmax) {
                //                        E5j1j2 += 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                //                                * ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
                //                        E5j1j2 += 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                //                                * ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
                //                        E5j1k1 += 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                //                                * ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
                //                        E5j2k2 += 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                //                                * ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
                //                        E5j1j2 -= 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1]
                //                                * ~f[j2][m] * f[i][n] * f[j1][m] * f[j2][m - 1];
                //                        E5j1j2 -= 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1]
                //                                * ~f[j1][m] * f[i][n] * f[j2][m] * f[j1][m - 1];
                //                        E5j1k1 -= 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m]
                //                                * ~f[k1][n] * f[i][n] * f[j1][m] * f[k1][n + 1];
                //                        E5j2k2 -= 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                //                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m]
                //                                * ~f[k2][n] * f[i][n] * f[j2][m] * f[k2][n + 1];
                //                    }
                //                }
            }

        }
//        cout << Ec << endl;

        Ec += E0 / norm2[i];

        Ec += E1j1 / (norm2[i] * norm2[j1]);
        Ec += E1j2 / (norm2[i] * norm2[j2]);

        Ec += E2j1 / (norm2[i] * norm2[j1]);
        Ec += E2j2 / (norm2[i] * norm2[j2]);

        Ec += E3j1 / (norm2[i] * norm2[j1]);
        Ec += E3j2 / (norm2[i] * norm2[j2]);

        Ec += (E4j1j2 + ~E4j1j2) / (norm2[i] * norm2[j1] * norm2[j2]);
        //        Ec += E4j1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
        //        Ec += E4j2k2 / (norm2[i] * norm2[j2] * norm2[k2]);

//        //        Ec += E5j1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
        Ec += (E5j1k1 + ~E5j1k1) / (norm2[i] * norm2[j1] * norm2[k1]);
        Ec += (E5j2k2 + ~E5j2k2) / (norm2[i] * norm2[j2] * norm2[k2]);
    }

//        cout << Ec << endl;
    En[0] = Ec.real();
    //    static int count = 0;
    //    cout << count++ << endl;

    //#include "vars.cpp"
    //#include "return.cpp"
    //    
    //    En[0] = E;
}

void energyprob::objfun_impl3(fitness_vector& En, const decision_vector& x) const {

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
            E0 += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

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
    }

    En[0] = Ec.real();
    //    static int count = 0;
    //    cout << count++ << endl;

    //#include "vars.cpp"
    //#include "return.cpp"
    //    
    //    En[0] = E;
}
